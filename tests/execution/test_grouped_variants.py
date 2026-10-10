import copy

import numpy as np
import pytest
from ase import Atoms
from ase.io import read, write

from gdpx.cli.compute import run_computation
from gdpx.execution.factory import create_worker, create_workers
from gdpx.execution.lifecycle import collect_compute, create_runtime_workers, prepare_compute, submit_compute
from gdpx.providers import DispatchConfig


def swept_config(scheduler="direct"):
    return {
        "potential": {"provider": "emt"},
        "executor": {
            "provider": "ase", "method": "md",
            "parameters": {"ensemble": "nvt", "steps": 2, "dump_period": 1, "random_seed": 17},
            "broadcast": {"temp": [400, 500, 600, 700]},
        },
        "scheduler": {"provider": scheduler, "parameters": {}},
        "dispatch": {"worker": "batch", "batch_size": 4},
    }


def atoms():
    return Atoms("Cu2", positions=[[0, 0, 0], [2.5, 0, 0]], cell=[8, 8, 8], pbc=True)


def test_batching_is_automatic_and_needs_no_variant_option():
    config = swept_config()
    workers = create_runtime_workers(config)
    assert len(workers) == 1 and len(workers[0].drivers) == 4
    assert "group_variants" not in DispatchConfig().to_dict()
    for legacy in (False, True):
        config["dispatch"]["group_variants"] = legacy
        assert len(create_runtime_workers(config)) == 1


def test_grouped_broadcast_plans_one_concurrent_slurm_batch(tmp_path):
    config = swept_config("slurm")
    config["scheduler"]["parameters"].update(concurrent_tasks=4, ntasks=4, **{"gpus-per-task": 1})
    original = copy.deepcopy(config)
    plan = prepare_compute(config, [atoms()], tmp_path)
    assert config == original
    assert len(plan.workers) == len(plan.workers[0].batches) == 1
    tasks = plan.workers[0].batches[0].tasks
    assert [task.driver_index for task in tasks] == [0, 1, 2, 3]
    assert [task.structure_index for task in tasks] == [0] * 4
    scripts = list((tmp_path / "_meta/jobscripts").glob("run-*.script"))
    assert len(scripts) == 1
    script = scripts[0].read_text()
    assert "task_count=4" in script and "concurrent_tasks=4" in script
    assert 'compute run --job' in script and '--task "$task"' in script


def test_grouped_saved_tasks_rehydrate_the_correct_executor(tmp_path):
    plan = prepare_compute(swept_config(), [atoms()], tmp_path)
    worker = create_runtime_workers(plan.config)[0]
    worker.directory = tmp_path
    manifest = worker.metadata.inputs.read()
    uid = next(iter(manifest["jobs"]))
    restored = create_worker(manifest["jobs"][uid]["input"]["runtime"], directory=tmp_path)
    assert [item.config.executor.parameters["temp"] for item in restored.runtimes] == [400, 500, 600, 700]
    for index, temp in enumerate([400, 500, 600, 700]):
        run_computation(["run"], None, directory=tmp_path, job=uid, task=index)
        frame = read(tmp_path / f"cand{index}/traj.xyz", index=0)
        assert frame.get_temperature() == pytest.approx(temp, rel=1e-5)
    # Immutable inputs and per-task seeds survive reconstruction.
    assert prepare_compute(swept_config(), [atoms()], tmp_path).plan_id == plan.plan_id


def test_grouped_multiple_structures_preserve_task_mapping_and_seed_policy(tmp_path):
    worker = create_runtime_workers(swept_config())[0]
    worker.directory = tmp_path
    _, frames, batches = worker.prepare_batches([atoms(), atoms()], persist=False)
    assert worker._task_plan == [(di, si) for si in range(2) for di in range(4)]
    assert len(frames) == 2
    assert len(batches) == 1  # direct execution groups every task
    expected = np.random.Generator(np.random.PCG64(17)).integers(0, 1e8, size=2)
    assert batches[0][3] == [int(seed) for seed in expected for _ in range(4)]


def test_grouped_lifecycle_collects_every_variant(tmp_path):
    plan = prepare_compute(swept_config(), [atoms()], tmp_path)
    submit_compute(plan)
    assert collect_compute(plan).number_of_trajectories == 4


def test_compute_cli_accepts_grouped_executor_broadcast(tmp_path):
    source = tmp_path / "ini.xyz"
    write(source, atoms())
    result = run_computation([str(source)], swept_config(), directory=tmp_path / "compute")
    assert result.state == "finished" and result.total == 1
    assert collect_compute(tmp_path / "compute").number_of_trajectories == 4


@pytest.mark.parametrize("field", ["scheduler", "potential", "dispatch", "executor"])
def test_incompatible_groups_are_rejected(field):
    runtimes = create_runtime_workers(swept_config())[0].as_dict()
    if field == "scheduler":
        runtimes[1][field]["parameters"]["concurrent_tasks"] = 2
    elif field == "potential":
        runtimes[1][field]["parameters"]["label"] = "different"
    elif field == "dispatch":
        runtimes[1][field]["batch_size"] = 2
    else:
        runtimes[1][field]["method"] = "spc"
    with pytest.raises(ValueError, match="identical|same executor"):
        create_worker(runtimes)


def test_grouped_implicit_seeds_resume_and_explicit_changes_conflict(tmp_path):
    config = swept_config()
    del config["executor"]["parameters"]["random_seed"]
    worker = create_runtime_workers(config)[0]
    worker.directory = tmp_path
    worker.prepare_batches([atoms()])
    catalog = (tmp_path / "_meta/inputs.json").read_bytes()
    restored = create_runtime_workers(config)[0]
    restored.directory = tmp_path
    restored.prepare_batches([atoms()])
    assert (tmp_path / "_meta/inputs.json").read_bytes() == catalog
    configs = restored.as_dict()
    configs[-1]["executor"]["parameters"]["random_seed"] = 42
    with pytest.raises(ValueError, match="all specify random_seed"):
        create_worker(configs)


def test_batch_size_counts_simulations_across_structures_and_variants(tmp_path):
    config = swept_config("slurm")
    config["dispatch"]["batch_size"] = 3
    plan = prepare_compute(config, [atoms(), atoms()], tmp_path)
    assert len(plan.workers) == 1
    batches = plan.workers[0].batches
    assert [len(batch.tasks) for batch in batches] == [3, 3, 2]
    assert [task.workdir for batch in batches for task in batch.tasks] == [f"cand{i}" for i in range(8)]
    assert plan.workers[0].directory == "."
    assert not (tmp_path / "w0").exists()


def test_incompatible_policies_choose_worker_subfolders_automatically(tmp_path):
    configs = create_runtime_workers(swept_config())[0].as_dict()[:3]
    configs[1]["dispatch"]["batch_size"] = 2
    workers = create_workers(configs, directory=tmp_path)
    assert [worker.directory.name for worker in workers] == ["w0", "w1"]
    assert [runtime.config.executor.parameters["temp"] for runtime in workers[0].runtimes] == [400, 600]
    assert workers[1].runtime.config.executor.parameters["temp"] == 500
    plan = prepare_compute(configs, [atoms()], tmp_path)
    assert [worker.directory for worker in plan.workers] == ["w0", "w1"]
    submit_compute(plan)
    assert collect_compute(plan).number_of_trajectories == 3
    assert (tmp_path / "w0/cand1").exists() and (tmp_path / "w1/cand0").exists()


def test_existing_plan_keeps_its_frozen_independent_worker_layout(tmp_path, monkeypatch):
    from gdpx.execution.lifecycle import service

    configs = create_runtime_workers(swept_config())[0].as_dict()[:2]

    def old_factory(values):
        return [create_worker(value) for value in values]

    with monkeypatch.context() as patch:
        patch.setattr(service, "create_workers", old_factory)
        plan = prepare_compute(configs, [atoms()], tmp_path)
    assert [worker.directory for worker in plan.workers] == ["w0", "w1"]
    assert len(create_workers(configs)) == 1
    submit_compute(plan)
    assert collect_compute(plan).number_of_trajectories == 2
    assert (tmp_path / "w0/cand0").exists() and (tmp_path / "w1/cand0").exists()
    assert not (tmp_path / "cand0").exists()


def test_historical_variant_flag_preserves_saved_fingerprints(tmp_path):
    config = swept_config()
    config["dispatch"]["group_variants"] = True
    plan = prepare_compute(config, [atoms()], tmp_path)
    submit_compute(plan)
    assert collect_compute(plan).number_of_trajectories == 4
