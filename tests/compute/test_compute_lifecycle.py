import copy
import json
import shutil

from ase import Atoms

from gdpx.execution.lifecycle import (
    PlanConflictError,
    collect_compute,
    inspect_compute,
    load_compute_plan,
    prepare_compute,
    submit_compute,
    run_compute_batch,
)


def _emt_config():
    return {
        "schema_version": 4,
        "potential": {"provider": "emt", "parameters": {}},
        "executor": {
            "provider": "ase",
            "method": "min",
            "parameters": {"dump_period": 1, "steps": 1, "fmax": 0.5, "random_seed": 17},
        },
        "dispatch": {},
    }


def _cu():
    return Atoms("Cu", positions=[[0.0, 0.0, 0.0]], cell=[4.0, 4.0, 4.0], pbc=True)


def test_prepare_is_immutable_and_does_not_submit(tmp_path):
    config = _emt_config()
    original = copy.deepcopy(config)

    plan = prepare_compute(config, [_cu()], tmp_path)

    assert config == original
    assert plan.config["potential"] == original["potential"]
    assert plan.schema_version == 6
    assert plan.path.exists()
    assert not json.loads((tmp_path / "_meta" / "scheduler.json").read_text())["_default"]
    assert len(list((tmp_path / "_meta" / "jobscripts").glob("run-*.script"))) == 1

    loaded = load_compute_plan(tmp_path)
    assert loaded.plan_id == plan.plan_id
    assert prepare_compute(config, [_cu()], tmp_path).plan_id == plan.plan_id


def test_prepare_rejects_a_different_plan_in_same_directory(tmp_path):
    prepare_compute(_emt_config(), [_cu()], tmp_path)
    changed = _cu()
    changed.positions[0, 0] = 0.2

    try:
        prepare_compute(_emt_config(), [changed], tmp_path)
    except PlanConflictError:
        pass
    else:
        raise AssertionError("different input should conflict with the immutable plan")


def test_local_submit_status_and_collect_round_trip(tmp_path):
    plan = prepare_compute(_emt_config(), [_cu()], tmp_path)

    submission = submit_compute(plan)
    assert submission.submitted_batches == ("w0/b0",)
    assert submit_compute(plan).submitted_batches == ()

    status = inspect_compute(plan)
    assert status.state == "finished"
    assert status.total == 1

    result = collect_compute(plan)
    assert result.number_of_trajectories == 1
    assert (tmp_path / "results" / "end_frames.xyz").exists()

    records = json.loads((tmp_path / "_meta" / "scheduler.json").read_text())["_default"]
    record = next(iter(records.values()))
    assert record["scheduler_job_id"] == "direct"
    assert record["attempt"] == 1


def test_submit_all_dry_run_scheduler_batches(tmp_path):
    config = _emt_config()
    config["scheduler"] = {"provider": "slurm", "parameters": {"is_dry_run": True}}
    config["dispatch"] = {"batch_size": 1}
    second = _cu()
    second.positions[0, 0] = 0.1
    plan = prepare_compute(config, [_cu(), second], tmp_path)

    result = submit_compute(plan)

    assert result.submitted_batches == ("w0/b0", "w0/b1")
    records = json.loads((tmp_path / "_meta" / "scheduler.json").read_text())["_default"]
    assert len(records) == 2
    assert {record["group_number"] for record in records.values()} == {0, 1}


def test_staged_plan_runs_in_its_new_working_tree(tmp_path):
    source = tmp_path / "source"
    staged = tmp_path / "staged"
    original = prepare_compute(_emt_config(), [_cu()], source)
    shutil.copytree(source, staged)

    loaded = load_compute_plan(staged)
    assert loaded.directory == str(staged.resolve())
    assert loaded.plan_id == original.plan_id
    assert run_compute_batch(loaded, batch=0).finished
    assert (staged / "cand0").exists()
    assert not (source / "cand0").exists()


def test_spawned_cli_runs_batch_without_submitting_again(tmp_path):
    from ase.io import write
    from gdpx.cli.compute import run_computation

    inputs = tmp_path / "input.xyz"
    write(inputs, [_cu()])
    run_computation(
        [str(inputs)], _emt_config(), batch=0, spawn=True, directory=tmp_path
    )
    assert (tmp_path / "cand0").exists()
    assert not json.loads((tmp_path / "_meta" / "scheduler.json").read_text())["_default"]


def test_queued_plan_script_survives_staging_and_pbs_launch(tmp_path):
    import os
    import subprocess

    config = _emt_config()
    config["scheduler"] = {"provider": "pbs", "parameters": {"is_dry_run": True}}
    source = tmp_path / "source"
    staged = tmp_path / "staged with spaces"
    plan = prepare_compute(config, [_cu()], source)
    submit_compute(plan)
    script = next((source / "_meta" / "jobscripts").glob("run-*.script"))
    script.write_text(script.read_text().replace(
        "cd ", 'gdp() { pwd > launch-cwd; printf "%s\\n" "$@" > launch-args; }\ncd ', 1
    ))
    shutil.copytree(source, staged)
    script = staged / "_meta" / "jobscripts" / script.name
    subprocess.run(["bash", str(script)], cwd=tmp_path,
                   env=dict(os.environ, PBS_O_WORKDIR=str(script.parent)), check=True)
    assert (staged / "launch-cwd").read_text().strip() == str(staged)
    args = (staged / "launch-args").read_text().splitlines()
    assert args[:3] == ["compute", "run", "--job"]
    assert args[3] in json.loads((staged / "_meta" / "inputs.json").read_text())["jobs"]


def test_shared_workdir_cache_is_retrievable_after_restart(tmp_path):
    config = _emt_config()
    config["dispatch"] = {"share_workdir": True}
    plan = prepare_compute(config, [_cu()], tmp_path)
    submit_compute(plan)
    assert json.loads((tmp_path / "_meta" / "scheduler.json").read_text())["results"]
    assert inspect_compute(load_compute_plan(tmp_path)).state == "finished"
    assert collect_compute(load_compute_plan(tmp_path)).number_of_trajectories == 1
    assert not (tmp_path / "_data").exists()


def test_canonical_inputs_are_shared_and_lossless(tmp_path):
    import numpy as np
    import pytest
    from ase.constraints import FixAtoms
    from gdpx.execution.fingerprint import read_structure_inputs, structure_digest

    atoms = _cu()
    atoms.positions[0, 0] = 0.12345678901234567
    atoms.set_constraint(FixAtoms(indices=[0]))
    atoms.set_initial_charges([0.25])
    atoms.set_initial_magnetic_moments([1])
    atoms.set_momenta([[0.1, 0.2, 0.3]], apply_constraint=False)
    plan = prepare_compute(_emt_config(), [atoms], tmp_path)
    assert plan.structure_digest == plan.workers[0].structure_digest == structure_digest([atoms])
    restored = read_structure_inputs(tmp_path / plan.structure_file, plan.structure_digest)
    assert np.array_equal(restored[0].positions, atoms.positions)
    assert len(restored[0].constraints) == 1
    atoms.positions[0, 0] = np.nextafter(atoms.positions[0, 0], np.inf)
    with pytest.raises(PlanConflictError):
        prepare_compute(_emt_config(), [atoms], tmp_path)


def test_modified_snapshot_and_plan_are_rejected_before_submission(tmp_path):
    import dataclasses
    import pytest
    from ase.io.jsonio import decode, encode
    from gdpx.execution.lifecycle.service import ComputeLifecycleError

    plan = prepare_compute(_emt_config(), [_cu()], tmp_path)
    with pytest.raises(ComputeLifecycleError, match="fingerprint mismatch"):
        submit_compute(dataclasses.replace(plan, structure_digest="modified"))
    path = tmp_path / plan.structure_file
    data = decode(path.read_text())
    data["structures"][plan.structure_digest][0].positions[0, 0] = 0.01
    path.write_text(encode(data))
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        submit_compute(plan)
    assert not json.loads((tmp_path / "_meta" / "scheduler.json").read_text())["_default"]


def test_saved_job_cli_uses_staged_snapshot_without_resubmitting(tmp_path):
    from gdpx.execution.factory import create_worker
    from gdpx.cli.compute import run_computation
    from ase.io.jsonio import decode

    config = _emt_config()
    config["scheduler"] = {"provider": "slurm", "parameters": {"is_dry_run": True}}
    source = tmp_path / "source"
    staged = tmp_path / "staged"
    worker = create_worker(config, directory=source)
    worker.run([_cu()], rng_states=[876])
    uid, saved = next(iter(decode((source / "_meta" / "inputs.json").read_text())["jobs"].items()))
    assert saved["input"]["random_seeds"] == [876]
    shutil.copytree(source, staged)
    records = staged / "_meta" / "scheduler.json"
    before = records.read_bytes()
    run_computation(["run"], None, job=uid, directory=staged)
    assert records.read_bytes() == before
    assert (staged / "cand0").exists()
    assert not (source / "cand0").exists()


def test_old_plan_schema_is_rejected_before_parsing_workers(tmp_path):
    import pytest
    from gdpx.execution.lifecycle.service import ComputeLifecycleError

    path = tmp_path / "_meta" / "compute-plan.json"
    path.parent.mkdir()
    path.write_text(json.dumps({"schema_version": 3, "workers": [{"identifier": "old-md5"}]}))
    before = path.read_bytes()
    with pytest.raises(ComputeLifecycleError, match="Unsupported compute plan schema"):
        load_compute_plan(tmp_path)
    assert path.read_bytes() == before


def test_saved_job_corruption_is_rejected_before_resolving_runtime(tmp_path, monkeypatch):
    import pytest
    from gdpx.cli.compute import run_computation
    from gdpx.execution import factory

    from ase.io.jsonio import decode, encode

    plan = prepare_compute(_emt_config(), [_cu()], tmp_path)
    path = plan.path
    data = decode(path.read_text())
    uid = next(iter(data["jobs"]))
    data["jobs"][uid]["job_digest"] = "invalid"
    path.write_text(encode(data))
    monkeypatch.setattr(factory, "create_worker", lambda *a, **k: pytest.fail("resolved corrupt input"))
    with pytest.raises(ValueError, match="Job fingerprint mismatch"):
        run_computation(["run"], None, job=uid, directory=tmp_path)
