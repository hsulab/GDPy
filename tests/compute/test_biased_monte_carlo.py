"""Relaxed endpoint acceptance and the biased-only evaluator/filter contract."""
import copy

import numpy as np
import pytest
from ase.constraints import FixAtoms
from ase.io import read

from gdpx.execution import resolve_runtime
from gdpx.execution.executor import RunState
from gdpx.execution.factory import create_worker
from gdpx.exploration.factory import create_exploration
from gdpx.sampling.acceptance import AcceptanceRule
from test_sampling_config import sampling_config
from test_sampling_lifecycle import copper


def config():
    return sampling_config("biased_monte_carlo")


def test_defaults_optimizer_scheduler_and_roundtrip():
    source = config()
    setup = source["executor"]["parameters"]["setup"]
    setup["operators"][0].pop("skip_distance_check")
    setup["evaluation"]["optimizer"] = {"name": "bfgs", "parameters": {"maxstep": .1}}
    setup["evaluation"]["scheduler"] = {"provider": "pbs"}
    source["scheduler"] = {"provider": "slurm"}
    original = copy.deepcopy(source)
    runtime = resolve_runtime(source)
    engine = runtime.executor
    assert engine.name == engine.setting.task == "biased_monte_carlo"
    assert runtime.scheduler.name == "slurm"
    assert engine.worker.scheduler.name == "pbs"
    assert engine.worker.runtime.config.executor.provider == "ase"
    assert engine.worker.runtime.config.executor.method == "min"
    assert engine.worker.driver.setting.driver_cls.keywords["maxstep"] == .1
    assert not engine.operators[0].skip_distance_check
    assert engine.worker.runtime.config.potential == runtime.config.potential
    assert resolve_runtime(engine.as_dict()).executor.as_dict() == engine.as_dict()
    assert source == original


@pytest.mark.parametrize("method", ["spc", "md", "cmin"])
def test_rejects_non_min_evaluator(method):
    source = config()
    source["executor"]["parameters"]["setup"]["evaluation"]["method"] = method
    with pytest.raises(ValueError, match="Unsupported evaluation fields: method"):
        resolve_runtime(source)
    engine = resolve_runtime(config()).executor
    # Registration also enforces the method even when bypassing the factory.
    worker = create_worker({"potential": {"provider": "emt"},
                           "executor": {"provider": "ase", "method": "spc"}})
    with pytest.raises(ValueError, match="requires runtime.executor.method: min"):
        engine.register_worker(worker)


@pytest.mark.parametrize("stop", [None, {}, {"steps": 0, "fmax": .1},
    {"steps": True, "fmax": .1}, {"steps": 1.5, "fmax": .1},
    {"steps": 10, "fmax": 0}, {"steps": 10, "fmax": True},
    {"steps": 10, "fmax": float("nan")}, {"steps": 10, "fmax": float("inf")}])
def test_requires_explicit_valid_minimisation_limits(stop):
    source = config()
    source["executor"]["parameters"]["setup"]["evaluation"]["stop"] = stop
    with pytest.raises((ValueError, TypeError)):
        resolve_runtime(source)


@pytest.mark.parametrize("field", ["potential", "modifiers"])
def test_evaluator_cannot_override_surface(field):
    source = config()
    source["executor"]["parameters"]["setup"]["evaluation"][field] = {}
    with pytest.raises(ValueError, match="Unsupported evaluation fields"):
        resolve_runtime(source)


@pytest.mark.parametrize("ensemble", ["nvt", "uvt", "suvt"])
def test_shared_ensemble_and_explicit_provider(ensemble):
    source = config()
    params = source["executor"]["parameters"]
    params["ensemble"]["method"] = ensemble
    if ensemble != "nvt":
        params["ensemble"]["chemical_potentials"] = {"Cu": 0., "Ni": .2}
        params["setup"]["operators"] = [{"method": "swap_type", "particles": ["Cu", "Ni"]}]
    params["setup"]["evaluation"].update(provider="ase")
    runtime = resolve_runtime(source)
    assert runtime.executor.operators[0].temperature == 900
    if ensemble != "nvt":
        assert runtime.executor.operators[0].chempots == [0., .2]


def test_hmc_and_conventional_mc_keep_restrictions():
    for method in ("monte_carlo", "hybrid_monte_carlo"):
        source = sampling_config(method)
        params = source["executor"]["parameters"]
        mc = params["setup"][1]["parameters"] if method == "hybrid_monte_carlo" else params
        mc["setup"]["operators"][0]["skip_distance_check"] = False
        with pytest.raises(ValueError, match="skip_distance_check"):
            resolve_runtime(source)
    source = sampling_config("hybrid_monte_carlo")
    child = config()["executor"]
    child["parameters"].pop("ensemble")
    source["executor"]["parameters"]["setup"][1] = child
    with pytest.raises(ValueError, match="conventional Monte Carlo children only"):
        resolve_runtime(source)
    with pytest.raises(ValueError, match="moved to gdp compute"):
        create_exploration({"method": "biased_monte_carlo"})


@pytest.mark.parametrize("accept", [True, False])
def test_acceptance_uses_relaxed_endpoints(tmp_path, monkeypatch, accept):
    source = config()
    params = source["executor"]["parameters"]
    params["stop"]["steps"] = 1
    params["setup"]["ignore_atoms_tags"] = False
    engine = resolve_runtime(source).executor
    engine.directory = tmp_path
    initial = copper()
    initial.set_tags([11, 22])
    initial.set_constraint(FixAtoms(indices=[0]))
    calls = []

    def decide(self, metadata, previous_energy, trial_energy, rng):
        calls.append((previous_energy, trial_energy))
        return accept

    monkeypatch.setattr(AcceptanceRule, "accept", decide)
    assert engine.run(initial) is RunState.FINISHED
    start = read(tmp_path / "steps/step.0000/cand0/traj.xyz", -1)
    trial_frames = read(tmp_path / "steps/step.0001/cand0/traj.xyz", ":")
    trial = trial_frames[-1]
    assert not np.allclose(initial.positions, start.positions)
    assert not np.allclose(trial_frames[0].positions, trial.positions)
    assert calls == [(start.get_potential_energy(), trial.get_potential_energy())]
    final = engine.read_trajectory()[-1]
    np.testing.assert_allclose(final.positions, (trial if accept else start).positions)
    np.testing.assert_array_equal(final.get_tags(), [11, 22])
    assert final.constraints[0].get_indices().tolist() == [0]
    assert len(engine.read_trajectory()) == 2


def test_filtered_proposals_consume_steps_without_minimisation(tmp_path):
    source = config()
    op = source["executor"]["parameters"]["setup"]["operators"][0]
    op.update(skip_distance_check=False, covalent_ratio=[10., 20.], max_random_attempts=1)
    engine = resolve_runtime(source).executor
    engine.directory = tmp_path
    assert engine.run(copper()) is RunState.FINISHED
    trajectory = engine.read_trajectory()
    assert len(trajectory) == 3
    for frame in trajectory[1:]:
        np.testing.assert_array_equal(frame.positions, trajectory[0].positions)
    assert not (tmp_path / "steps/step.0001").exists()
    assert not (tmp_path / "steps/step.0002").exists()
    assert len((tmp_path / "opstat.txt").read_text().splitlines()) == 3


def test_iteration_limit_endpoint_is_used(tmp_path, monkeypatch):
    source = config()
    params = source["executor"]["parameters"]
    params["stop"]["steps"] = 1
    params["setup"]["evaluation"]["stop"] = {"steps": 1, "fmax": 1e-12}
    monkeypatch.setattr(AcceptanceRule, "accept", lambda *args: True)
    engine = resolve_runtime(source).executor
    engine.directory = tmp_path
    assert engine.run(copper()) is RunState.FINISHED
    final = engine.read_trajectory()[-1]
    assert np.max(np.abs(final.get_forces())) > 1e-12
    trial = read(tmp_path / "steps/step.0001/cand0/traj.xyz", -1)
    np.testing.assert_array_equal(final.positions, trial.positions)


def test_alternate_provider_and_modifier_share_outer_surface():
    from dataclasses import replace
    from ase.calculators.emt import EMT
    from gdpx.providers import CapabilityKind, Provider, ProviderManager
    from gdpx.providers.builtin import register_builtin_providers
    from gdpx.providers.ase import ASE_PROVIDER

    class BiasFactory:
        def create(self, parameters, **kwargs):
            return EMT()

    manager = ProviderManager()
    register_builtin_providers(manager)
    manager.register(replace(ASE_PROVIDER, name="alternate_ase"))
    manager.register(Provider("test_bias", capabilities={CapabilityKind.MODIFIER: {"default": BiasFactory()}}))
    source = config()
    source["executor"]["parameters"]["setup"]["evaluation"]["provider"] = "alternate_ase"
    source["modifiers"] = [{"provider": "test_bias"}]
    runtime = resolve_runtime(source, providers=manager)
    child = runtime.executor.worker.runtime
    assert child.config.executor.provider == "alternate_ase"
    assert child.config.potential == runtime.config.potential
    assert child.config.modifiers == runtime.config.modifiers
    assert len(child.materialization.calculator.mixer.calcs) == 2
