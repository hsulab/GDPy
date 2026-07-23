import copy
import json

from ase import Atoms

from gdpx.compute import (
    PlanConflictError,
    collect_compute,
    inspect_compute,
    load_compute_plan,
    prepare_compute,
    submit_compute,
)


def _emt_config():
    return {
        "potential": {"name": "emt", "params": {"backend": "ase"}},
        "driver": {
            "task": "min",
            "backend": "ase",
            "init": {"dump_period": 1},
            "run": {"steps": 1, "fmax": 0.5},
            "random_seed": 17,
        },
    }


def _cu():
    return Atoms("Cu", positions=[[0.0, 0.0, 0.0]], cell=[4.0, 4.0, 4.0], pbc=True)


def test_prepare_is_immutable_and_does_not_submit(tmp_path):
    config = _emt_config()
    original = copy.deepcopy(config)

    plan = prepare_compute(config, [_cu()], tmp_path)

    assert config == original
    assert plan.config["potter"] == original["potential"]
    assert "potential" not in plan.config
    assert plan.path.exists()
    assert not (tmp_path / "_local_jobs.json").exists()
    assert (tmp_path / "_data" / "scripts" / "run-w0-b0.script").exists()

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

    records = json.loads((tmp_path / "_local_jobs.json").read_text())["_default"]
    record = next(iter(records.values()))
    assert record["scheduler_job_id"] == "local"
    assert record["attempt"] == 1


def test_submit_all_dry_run_scheduler_batches(tmp_path):
    config = _emt_config()
    config["scheduler"] = {"backend": "slurm", "is_dry_run": True}
    config["batchsize"] = 1
    second = _cu()
    second.positions[0, 0] = 0.1
    plan = prepare_compute(config, [_cu(), second], tmp_path)

    result = submit_compute(plan)

    assert result.submitted_batches == ("w0/b0", "w0/b1")
    records = json.loads((tmp_path / "_slurm_jobs.json").read_text())["_default"]
    assert len(records) == 2
    assert {record["group_number"] for record in records.values()} == {0, 1}
