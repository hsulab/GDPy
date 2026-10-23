"""Independent sampling chains, deferred evaluations, and durable completion."""
import json

import numpy as np
import pytest
from ase import Atoms
from ase.io import read

from gdpx.execution import resolve_runtime
from gdpx.execution.executor import RunState
from gdpx.execution.lifecycle import (
    collect_compute, inspect_compute, prepare_compute, resubmit_compute, submit_compute,
)
from gdpx.execution.lifecycle.service import orchestrate_compute
from gdpx.execution.schedulers.direct import DirectScheduler
from gdpx.execution.checkpoint import load_data
from test_sampling_config import sampling_config


def copper():
    return Atoms("Cu2", positions=[[3, 3, 3], [5.5, 3, 3]], cell=[12] * 3)


@pytest.mark.parametrize("method", ["monte_carlo", "biased_monte_carlo", "hybrid_monte_carlo"])
@pytest.mark.parametrize("steps,period,expected", [(0, 10, 1), (3, 10, 2), (3, 2, 3)])
def test_multiple_chains_sparse_output_and_collection(tmp_path, method, steps, period, expected):
    config = sampling_config(method)
    params = config["executor"]["parameters"]
    params["stop"]["steps"] = steps
    params["output"] = {"trajectory": {"period": period}, "checkpoint": {"period": 100}}
    second = copper()
    second.positions[1, 0] = 5.7
    plan = prepare_compute(config, [copper(), second], tmp_path)
    assert inspect_compute(plan).state == "prepared"
    assert orchestrate_compute(plan).state == "finished"
    for index in (0, 1):
        chain = tmp_path / f"cand{index}"
        assert len(read(chain / "mc.xyz", ":")) == expected
        state = load_data(chain / f"checkpoint.{steps}" / "state.json")
        assert state["step"] == steps
    assert len(read(tmp_path / "results/end_frames.xyz", ":")) == 2
    before = [(tmp_path / f"cand{i}/mc.xyz").read_bytes() for i in (0, 1)]
    assert orchestrate_compute(plan).state == "finished"
    assert before == [(tmp_path / f"cand{i}/mc.xyz").read_bytes() for i in (0, 1)]
    collect_compute(plan, archive=True)
    # Collection remains possible after candidate directories are archived.
    assert collect_compute(plan).number_of_trajectories == 2


@pytest.mark.parametrize("method,delayed", [
    ("monte_carlo", "step.0000"),
    ("monte_carlo", "step.0001"),
    ("biased_monte_carlo", "step.0000"),
    ("biased_monte_carlo", "step.0001"),
    ("hybrid_monte_carlo", "procedure.0000"),
    ("hybrid_monte_carlo", "procedure.0001/cand0/steps/step.0001"),
])
def test_pending_child_resume_matches_uninterrupted(tmp_path, monkeypatch, method, delayed):
    config = sampling_config(method)
    initial = copper()
    if method == "biased_monte_carlo":
        from ase.constraints import FixAtoms
        initial.set_constraint(FixAtoms(indices=[0]))
        initial.set_tags([11, 22])
        config["executor"]["parameters"]["setup"]["ignore_atoms_tags"] = False
    baseline = prepare_compute(config, [initial], tmp_path / "baseline")
    orchestrate_compute(baseline)
    original_submit = DirectScheduler.submit
    original_finished = DirectScheduler.is_finished
    pending = {}
    delayed_once = set()
    submissions = []

    def submit(scheduler, func_to_execute=None):
        key = str(scheduler.script)
        if "deferred" in key and delayed in key:
            submissions.append(key)
            if key not in delayed_once:
                delayed_once.add(key)
                pending[key] = func_to_execute
                return "deferred"
        return original_submit(scheduler, func_to_execute)

    def finished(scheduler):
        return str(scheduler.script) not in pending and original_finished(scheduler)

    monkeypatch.setattr(DirectScheduler, "submit", submit)
    monkeypatch.setattr(DirectScheduler, "is_finished", finished)
    plan = prepare_compute(config, [initial], tmp_path / "deferred")
    submit_compute(plan)
    assert inspect_compute(plan).state == "running"
    before = list(submissions)
    inspect_compute(plan)
    assert submissions == before
    for _ in range(10):
        callbacks = list(pending.values())
        pending.clear()
        for callback in callbacks:
            callback()
        resubmit_compute(plan)
        if inspect_compute(plan).state == "finished":
            break
    assert inspect_compute(plan).state == "finished"
    assert len(submissions) == len(set(submissions)), "pending children were submitted twice"
    collect_compute(plan)
    expected = read(tmp_path / "baseline/cand0/mc.xyz", ":")
    actual = read(tmp_path / "deferred/cand0/mc.xyz", ":")
    assert len(expected) == len(actual)
    for first, second in zip(expected, actual):
        np.testing.assert_array_equal(first.positions, second.positions)
        assert first.get_potential_energy() == second.get_potential_energy()
        if method == "biased_monte_carlo":
            np.testing.assert_array_equal(second.get_tags(), [11, 22])
            assert second.constraints[0].get_indices().tolist() == [0]
    state = load_data(tmp_path / "deferred/cand0/checkpoint.2/state.json")
    baseline_state = load_data(tmp_path / "baseline/cand0/checkpoint.2/state.json")
    assert state["rng"] == baseline_state["rng"]


@pytest.mark.parametrize("method", ["monte_carlo", "biased_monte_carlo", "hybrid_monte_carlo"])
def test_checkpoint_resume_and_earlystop(tmp_path, method):
    config = sampling_config(method)
    params = config["executor"]["parameters"]
    params["stop"] = {"steps": 10, "earlystop": {"property": "energy_per_atom", "range": [-100, -99]}}
    params["output"] = {"trajectory": {"period": 20}}
    runtime = resolve_runtime(config)
    engine = runtime.executor
    engine.directory = tmp_path
    assert engine.run(copper()) is RunState.FINISHED
    assert len(engine.read_trajectory()) == 2
    # Publication is sufficient even if interrupted before writing the marker.
    (tmp_path / "MC_EARLY_STOPPED").unlink()
    restored = resolve_runtime(config).executor
    restored.directory = tmp_path
    before = (tmp_path / "mc.xyz").read_bytes()
    assert restored.run(copper()) is RunState.FINISHED
    assert (tmp_path / "mc.xyz").read_bytes() == before


def test_direct_executor_rejects_population(tmp_path):
    engine = resolve_runtime(sampling_config()).executor
    engine.directory = tmp_path
    with pytest.raises(TypeError, match="one Atoms"):
        engine.run([copper(), copper()])


def test_checkpoint_retention_and_read_only_fallback(tmp_path):
    config = sampling_config()
    config["executor"]["parameters"]["stop"]["steps"] = 3
    config["executor"]["parameters"]["output"] = {"checkpoint": {"period": 1, "keep": 3}}
    engine = resolve_runtime(config).executor
    engine.directory = tmp_path
    engine.run(copper())
    manifest = tmp_path / "current.json"
    original = manifest.read_bytes()
    assert len(json.loads(original)["snapshots"]) == 3
    (tmp_path / "checkpoint.3/state.json").write_text("{}")
    assert not engine.read_convergence()
    assert manifest.read_bytes() == original


def test_child_failure_does_not_complete_chain(tmp_path, monkeypatch):
    from gdpx.execution.workers.drive import DriverBatchError
    from gdpx.providers.ase.driver import AseDriver

    def fail(*args, **kwargs):
        raise RuntimeError("evaluation unavailable")

    plan = prepare_compute(sampling_config(), [copper()], tmp_path)
    with monkeypatch.context() as patch:
        patch.setattr(AseDriver, "_irun", fail)
        with pytest.raises(DriverBatchError, match="evaluation unavailable"):
            submit_compute(plan)
    assert not (tmp_path / "cand0/current.json").exists()
    assert not (tmp_path / "results/end_frames.xyz").exists()
    assert orchestrate_compute(plan).state == "finished"


@pytest.mark.parametrize("position", ["before", "after"])
def test_cli_runtime_argument_position(tmp_path, monkeypatch, position):
    import sys
    from gdpx.main import main
    from gdpx.cli import compute

    captured = []
    monkeypatch.setattr(compute, "run_computation", lambda *args, **kwargs: captured.append((args, kwargs)))
    command = ["gdp", "--log=", "-d", str(tmp_path)]
    if position == "before":
        command += ["--runtime", "mc.yaml"]
    command += ["compute", "initial.xyz"]
    if position == "after":
        command += ["--runtime", "mc.yaml"]
    monkeypatch.setattr(sys, "argv", command)
    assert main() is None
    assert captured[0][0] == (["initial.xyz"], "mc.yaml")
