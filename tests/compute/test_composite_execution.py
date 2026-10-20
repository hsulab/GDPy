"""Composite boundaries, state handoff, and fixed-surface validation."""
import copy
from pathlib import Path

import numpy as np
import pytest
import yaml
from ase.constraints import FixAtoms
from ase.io import read

from gdpx.execution import resolve_runtime
from gdpx.execution.executor import RunState
from gdpx.providers.configuration import expand_runtime_configs
from test_sampling_config import sampling_config
from test_sampling_lifecycle import copper


def annealing_config():
    path = Path(__file__).resolve().parents[2] / "examples/global_optimisation/runtimes/emt_sa.yaml"
    config = yaml.safe_load(path.read_text())
    config["executor"]["parameters"]["random_seed"] = 123
    setup = config["executor"]["parameters"]["setup"]
    setup[0]["parameters"]["stop"]["steps"] = 2
    setup[0]["parameters"]["output"]["trajectory"]["period"] = 1
    setup[1]["parameters"]["stop"] = {"steps": 100, "fmax": .2}
    return config


@pytest.mark.parametrize("method", ["hybrid_monte_carlo", "simulated_annealing"])
@pytest.mark.parametrize("boundary", ["child", "trajectory"])
def test_interrupted_boundary_matches_uninterrupted(tmp_path, monkeypatch, method, boundary):
    config = sampling_config(method) if method == "hybrid_monte_carlo" else annealing_config()
    baseline = resolve_runtime(config).executor
    baseline.directory = tmp_path / "baseline"
    assert baseline.run(copper()) is RunState.FINISHED
    engine = resolve_runtime(config).executor
    engine.directory = tmp_path / "resumed"
    original = engine._commit
    interrupted = False

    def commit(state, atoms, suffix=""):
        nonlocal interrupted
        selected = ((boundary == "child" and suffix == "-child.1") or
                    (boundary == "trajectory" and state["step"] == 1 and suffix == ""))
        if selected and not interrupted:
            interrupted = True
            raise RuntimeError("power loss before publication")
        return original(state, atoms, suffix)

    monkeypatch.setattr(engine, "_commit", commit)
    with pytest.raises(RuntimeError, match="power loss"):
        engine.run(copper())
    restored = resolve_runtime(config).executor
    restored.directory = engine.directory
    assert restored.run(copper()) is RunState.FINISHED
    for first, second in zip(baseline.read_trajectory(), restored.read_trajectory()):
        np.testing.assert_array_equal(first.positions, second.positions)
        np.testing.assert_array_equal(first.get_momenta(), second.get_momenta())
        assert first.get_potential_energy() == second.get_potential_energy()
    assert len(baseline.read_trajectory()) == len(restored.read_trajectory())
    before = (engine.directory / engine.TRAJ_NAME).read_bytes()
    restored.run(copper())
    assert (engine.directory / engine.TRAJ_NAME).read_bytes() == before


def test_repeated_children_have_distinct_state_and_handoff(tmp_path):
    config = sampling_config("hybrid_monte_carlo")
    params = config["executor"]["parameters"]
    params["setup"] *= 2
    # Preserve meaningful tags and incoming velocities across MC boundaries.
    for child in params["setup"]:
        if child["method"] == "monte_carlo":
            child["parameters"]["setup"]["ignore_atoms_tags"] = False
    initial = copper()
    initial.set_tags([4, 9])
    initial.set_constraint(FixAtoms(indices=[0]))
    initial.set_momenta([[0, 0, 0], [.2, .1, 0]])
    engine = resolve_runtime(config).executor
    engine.directory = tmp_path
    engine.run(initial)
    assert len(engine.read_trajectory()) == 3
    seeds = []
    for cycle in (1, 2):
        root = tmp_path / f"calculations/step.{cycle:04d}"
        for index in (1, 3):
            md_frames = read(root / f"procedure.{index-1:04d}/cand0/traj.xyz", ":")
            np.testing.assert_array_equal(md_frames[0].positions[0], md_frames[-1].positions[0])
            child = root / f"procedure.{index:04d}/cand0"
            assert len((child / "opstat.txt").read_text().splitlines()) == 3
            before = read(root / f"procedure.{index-1:04d}/cand0/traj.xyz", -1)
            after = read(child / "mc.xyz", 0)
            np.testing.assert_array_equal(before.get_momenta(), after.get_momenta())
        # Different invocations must not reuse the same Monte Carlo random stream.
        from gdpx.execution.workers.metadata import WorkerMetadata
        metadata = WorkerMetadata(root / "procedure.0001")
        definition = metadata.calculation_set(metadata.inputs.read(), ".")
        seeds.append(definition["batches"][0]["random_seeds"])
    assert seeds[0] != seeds[1]
    for frame in engine.read_trajectory():
        np.testing.assert_array_equal(frame.get_tags(), initial.get_tags())
        assert len(frame.constraints) == 1


@pytest.mark.parametrize("field,value", [("potential", {"provider": "emt"}),
                                         ("modifiers", []), ("runtime", {}), ("dispatch", {})])
def test_child_cannot_override_surface_or_dispatch(field, value):
    config = sampling_config("hybrid_monte_carlo")
    config["executor"]["parameters"]["setup"][0][field] = value
    with pytest.raises(ValueError, match="Unknown executor fields"):
        resolve_runtime(config)


def test_native_migration_default_and_single_point_override():
    config = sampling_config()
    assert resolve_runtime(config).executor.worker.runtime.config.executor.method == "spc"
    config["executor"]["parameters"]["setup"]["executor"] = {"method": "min"}
    with pytest.raises(ValueError, match="method: spc"):
        resolve_runtime(config)
    config["executor"]["provider"] = "gdpx"
    with pytest.raises(ValueError, match="renamed to native"):
        resolve_runtime(config)


def test_composite_broadcast_and_child_schedulers():
    config = sampling_config("hybrid_monte_carlo")
    config["executor"]["broadcast"] = {"setup.1.parameters.setup.ensemble.temperature": [300, 600]}
    config["scheduler"] = {"provider": "slurm"}
    config["executor"]["parameters"]["setup"][0]["scheduler"] = {"provider": "pbs"}
    variants = expand_runtime_configs(config)
    runtimes = [resolve_runtime(item) for item in variants]
    assert [r.executor.children[1].executor.operators[0].temperature for r in runtimes] == [300, 600]
    for runtime in runtimes:
        assert runtime.scheduler.name == "slurm"
        assert runtime.executor.children[0].scheduler.name == "pbs"
        assert runtime.executor.children[1].scheduler.name == "direct"
        assert all(child.config.potential == runtime.config.potential for child in runtime.executor.children)


@pytest.mark.parametrize("change", [
    lambda p: p.update(steps=3),
    lambda p: p["setup"].update(steps=3),
    lambda p: p["setup"].pop("ensemble"),
    lambda p: p["setup"]["ensemble"].update(method="bad"),
    lambda p: p["setup"]["operators"][0].update(temperature=1),
    lambda p: p["setup"]["operators"][0].update(skip_distance_check=False),
    lambda p: p["setup"].update(executor=None),
    lambda p: p.update(output={"checkpoint": {"period": 0}}),
    lambda p: p.update(output={"trajectory": {"period": True}}),
    lambda p: p.update(output={"checkpoint": {"keep": 0}}),
])
def test_invalid_mc_settings(change):
    config = sampling_config()
    change(config["executor"]["parameters"])
    with pytest.raises((ValueError, TypeError)):
        resolve_runtime(config)


def test_annealing_compute_lifecycle(tmp_path):
    from gdpx.execution.lifecycle import prepare_compute, collect_compute, inspect_compute
    from gdpx.execution.lifecycle.service import orchestrate_compute
    plan = prepare_compute(annealing_config(), [copper()], tmp_path)
    assert orchestrate_compute(plan).state == "finished"
    assert inspect_compute(plan).state == "finished"
    assert len(read(tmp_path / "cand0/annealing.xyz", ":")) == 2
    collect_compute(plan, archive=True)
    assert collect_compute(plan).number_of_trajectories == 1


@pytest.mark.parametrize("end", [None, True, 0, -10, float("nan"), 1300])
def test_annealing_rejects_invalid_temperature_ramps(end):
    config = annealing_config()
    config["executor"]["parameters"]["setup"][0]["parameters"]["setup"]["tend"] = end
    with pytest.raises((ValueError, TypeError)):
        resolve_runtime(config)


def test_incompatible_backend_fails_during_resolution():
    config = sampling_config("hybrid_monte_carlo")
    config["potential"]["backend"] = "not-a-backend"
    from gdpx.providers.errors import MaterializationError
    with pytest.raises(MaterializationError, match="backend"):
        resolve_runtime(config)


def test_different_execution_providers_share_surface_and_modifiers():
    from dataclasses import replace
    from ase.calculators.emt import EMT
    from gdpx.providers import ProviderManager, Provider, CapabilityKind
    from gdpx.providers.builtin import register_builtin_providers
    from gdpx.providers.ase import ASE_PROVIDER

    class BiasFactory:
        def create(self, parameters, **kwargs):
            return EMT()

    manager = ProviderManager()
    register_builtin_providers(manager)
    manager.register(replace(ASE_PROVIDER, name="alternate_ase"))
    manager.register(Provider("test_bias", capabilities={CapabilityKind.MODIFIER: {"default": BiasFactory()}}))
    config = sampling_config("hybrid_monte_carlo")
    config["executor"]["parameters"]["setup"][0]["provider"] = "alternate_ase"
    config["modifiers"] = [{"provider": "test_bias"}]
    original = copy.deepcopy(config)
    runtime = resolve_runtime(config, providers=manager)
    md, mc = runtime.executor.children
    assert config == original
    assert md.config.potential == mc.config.potential == runtime.config.potential
    assert md.config.modifiers == mc.config.modifiers == runtime.config.modifiers
    energy = mc.executor.worker.runtime
    assert md.materialization.calculator is not energy.materialization.calculator
    assert len(md.materialization.calculator.mixer.calcs) == 2
    assert len(energy.materialization.calculator.mixer.calcs) == 2


@pytest.mark.parametrize("stage", ["procedure.0000", "procedure.0001"])
def test_annealing_waits_for_child_and_resumes(tmp_path, monkeypatch, stage):
    from gdpx.execution.schedulers.direct import DirectScheduler
    config = annealing_config()
    expected = resolve_runtime(config).executor
    expected.directory = tmp_path / "baseline"
    expected.run(copper())
    original_submit = DirectScheduler.submit
    original_finished = DirectScheduler.is_finished
    pending, seen = {}, set()

    def submit(scheduler, func_to_execute=None):
        key = str(scheduler.script)
        if "deferred" in key and stage in key and key not in seen:
            seen.add(key)
            pending[key] = func_to_execute
            return "waiting"
        return original_submit(scheduler, func_to_execute)

    monkeypatch.setattr(DirectScheduler, "submit", submit)
    monkeypatch.setattr(DirectScheduler, "is_finished",
                        lambda scheduler: str(scheduler.script) not in pending and original_finished(scheduler))
    engine = resolve_runtime(config).executor
    engine.directory = tmp_path / "deferred"
    assert engine.run(copper()) is RunState.PENDING
    assert not engine.read_convergence()
    assert len(pending) == 1
    callbacks = list(pending.values())
    pending.clear()
    for callback in callbacks:
        callback()
    resumed = resolve_runtime(config).executor
    resumed.directory = engine.directory
    assert resumed.run(copper()) is RunState.FINISHED
    assert len(seen) == 1
    np.testing.assert_array_equal(expected.read_trajectory()[-1].positions,
                                  resumed.read_trajectory()[-1].positions)
