import json
import multiprocessing
import pathlib
import shutil
from types import SimpleNamespace

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io.jsonio import decode, encode

from gdpx.execution.factory import create_worker
from gdpx.execution.fingerprint import structure_digest
from gdpx.execution.lifecycle import (
    collect_compute, inspect_compute, load_compute_plan, prepare_compute, submit_compute,
)
from gdpx.execution.workers.metadata import Catalog, WorkerMetadata


def config(scheduler="direct"):
    return dict(potential=dict(provider="emt", parameters={}),
                executor=dict(provider="ase", method="spc", parameters=dict(random_seed=9)),
                scheduler=dict(provider=scheduler, parameters={"is_dry_run": True} if scheduler != "direct" else {}),
                dispatch={})


def atom(x=0.12345678901234567):
    return Atoms("Cu", positions=[[x, 0, 0]], cell=[4, 4, 4], pbc=True)


def test_multiple_workers_share_two_catalogs_and_prepared_scripts(tmp_path):
    configs = [config(), config()]
    plan = prepare_compute(configs, [atom()], tmp_path)
    inputs = tmp_path / "_meta" / "inputs.json"
    state = tmp_path / "_meta" / "scheduler.json"
    scripts = sorted((tmp_path / "_meta" / "jobscripts").iterdir())
    assert len(scripts) == 2
    assert set(tmp_path.glob("**/_meta/*.json")) == {inputs, state}
    before = inputs.read_bytes()
    data = decode(inputs.read_text())
    assert len(data["structures"]) == 1
    assert {job["worker"] for job in data["jobs"].values()} == {"w0", "w1"}
    assert not decode(state.read_text())["_default"]
    submit_compute(plan)
    assert inspect_compute(plan).total == 2
    assert collect_compute(plan).number_of_trajectories == 2
    assert sorted((tmp_path / "_meta" / "jobscripts").iterdir()) == scripts
    assert inputs.read_bytes() == before
    assert set(tmp_path.glob("**/_meta/*.json")) == {inputs, state}
    assert submit_compute(load_compute_plan(tmp_path)).submitted_batches == ()


def test_catalog_provenance_preserves_types_and_spaces(tmp_path):
    params = config()
    params["dispatch"]["retain_info"] = True
    worker = create_worker(params, directory=tmp_path)
    atoms = atom()
    atoms.info.update(confid=72, label="a label with spaces", scores=[1, 2], active=True)
    worker.run([atoms])
    output = worker.retrieve()[0][-1]
    assert output.info["confid"] == 72
    assert output.info["label"] == "a label with spaces"
    assert output.info["scores"] == [1, 2]
    assert output.info["active"] is True
    assert worker.metadata.frames(structure_digest([atoms]))[0].info == {}


def test_shared_results_preserve_calculator_data(tmp_path):
    params = config()
    params["dispatch"]["share_workdir"] = True
    worker = create_worker(params, directory=tmp_path)
    worker.run([atom(), atom(0.2)])
    frames = [trajectory[0] for trajectory in worker.retrieve()]
    assert len(frames) == 2
    assert all(np.isfinite(frame.get_potential_energy()) for frame in frames)
    assert all(frame.get_forces().shape == (1, 3) for frame in frames)
    assert {p.name for p in (tmp_path / "_meta").glob("*.json")} == {"inputs.json", "scheduler.json"}


def _update_catalog(path, worker):
    catalog = Catalog(path, "scheduler")
    for index in range(8):
        with catalog.transaction() as data:
            data["results"][f"{worker}-{index}"] = {}


def test_concurrent_updates_do_not_lose_records(tmp_path):
    path = tmp_path / "scheduler.json"
    context = multiprocessing.get_context("spawn")
    processes = [context.Process(target=_update_catalog, args=(path, index)) for index in range(4)]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=30)
        assert process.exitcode == 0
    assert len(Catalog(path, "scheduler").read()["results"]) == 32


@pytest.mark.parametrize("mode", ["worker", "plan", "inspect"])
@pytest.mark.parametrize("layout", ["legacy", "catalog"])
def test_previous_layout_requires_fresh_folder(tmp_path, mode, layout):
    metadata = tmp_path / "_meta"
    metadata.mkdir()
    if layout == "legacy":
        (metadata / "_scheduler.json").write_text('{"_default": {}}')
    else:
        (metadata / "inputs.json").write_text(encode(dict(
            format="gdpx-inputs", version=1, structures={}, workers={}, jobs={})))
    before = {p.name: p.read_bytes() for p in metadata.iterdir()}
    with pytest.raises(ValueError, match="new working directory"):
        if mode == "plan":
            prepare_compute(config(), [atom()], tmp_path)
        elif mode == "worker":
            create_worker(config(), directory=tmp_path).run([atom()])
        else:
            create_worker(config(), directory=tmp_path).inspect()
    assert {p.name: p.read_bytes() for p in metadata.iterdir()} == before


def test_corrupt_compact_catalog_does_not_fall_back_to_legacy(tmp_path):
    params = config()
    worker = create_worker(params, directory=tmp_path)
    worker.run([atom()])
    metadata = tmp_path / "_meta"
    (metadata / "_scheduler.json").write_text('{"_default": {}}')
    (metadata / "inputs.json").write_text('{"version": 999}')
    before = (metadata / "scheduler.json").read_bytes()
    with pytest.raises(ValueError, match="Invalid inputs catalog"):
        create_worker(params, directory=tmp_path).run([atom()])
    assert (metadata / "scheduler.json").read_bytes() == before


def test_remote_result_merge_protects_parent_and_sibling_jobs(tmp_path):
    from gdpx.execution.schedulers.remote import SshTransport
    from gdpx.execution.schedulers.direct import DirectScheduler

    local = tmp_path / "local"
    worker = create_worker(config("slurm"), directory=local)
    worker._share_wdir = True
    worker.run([atom()])
    job = worker.job_store.get_running()[0]
    remote = tmp_path / "remote" / job.gdir
    shutil.copytree(local, remote)
    remote_metadata = WorkerMetadata(remote)
    returned = atom()
    returned.info["wdir"] = "cand0"
    returned.calc = SinglePointCalculator(returned, energy=-2, forces=np.ones((1, 3)))
    remote_metadata.put_result(job.uid, returned)
    with remote_metadata.state.transaction() as data:
        data["_default"] = {}  # Remote lifecycle state must never replace the controller's.
        data["results"]["unrelated"] = {"ignored": {}}
    with worker.metadata.state.transaction() as data:
        data["results"]["sibling"] = {}
    before_inputs = worker.metadata.inputs.path.read_bytes()
    before_records = worker.metadata.state.read()["_default"]
    (local / "cand0").mkdir()
    (local / "cand0" / "obsolete").write_text("old")
    (remote / "cand0").mkdir()
    (remote / "cand0" / "result").write_text("new")

    class Sftp:
        def listdir_attr(self, directory):
            return [SimpleNamespace(filename=p.name, st_mode=p.stat().st_mode)
                    for p in pathlib.Path(directory).iterdir()]
        def lstat(self, path):
            return pathlib.Path(path).stat()
        stat = lstat
        def get(self, source, destination):
            shutil.copyfile(source, destination)
        def open(self, path, mode):
            return open(path, mode)
        def close(self):
            pass

    transport = SshTransport(DirectScheduler(), "test", str(remote.parent))
    transport._client = lambda: SimpleNamespace(open_sftp=lambda: Sftp(), close=lambda: None)
    worker.scheduler = transport
    worker._configure_scheduler_paths()
    transport.job_name = job.gdir
    transport.script = worker._script_path(job.uid)
    worker._sync_job(job)
    assert worker.metadata.inputs.path.read_bytes() == before_inputs
    assert worker.metadata.state.read()["_default"] == before_records
    assert "sibling" in worker.metadata.state.read()["results"]
    assert "unrelated" not in worker.metadata.state.read()["results"]
    assert worker.metadata.results(job.uid)[0].get_potential_energy() == -2
    assert not (local / "cand0" / "obsolete").exists()
    assert (local / "cand0" / "result").read_text() == "new"


def test_multiworker_staged_job_executes_only_its_worker(tmp_path):
    from gdpx.cli.compute import run_computation

    source, staged = tmp_path / "source", tmp_path / "staged"
    prepare_compute([config("slurm"), config("slurm")], [atom()], source)
    data = WorkerMetadata(source).inputs.read()
    uid = next(uid for uid, saved in data["jobs"].items() if saved["worker"] == "w1")
    shutil.copytree(source, staged)
    # Match actual SSH staging: the controller state and lock are not uploaded.
    (staged / "_meta" / "scheduler.json").unlink()
    (staged / "_meta" / ".metadata.lock").unlink()
    run_computation(["run"], None, directory=staged, job=uid)
    assert (staged / "w1" / "cand0").exists()
    assert not (staged / "w0" / "cand0").exists()
    assert not (source / "w1" / "cand0").exists()


def test_legacy_manifest_argument_requires_fresh_folder(tmp_path):
    from gdpx.cli.compute import run_computation

    manifest = tmp_path / "job-old.json"
    manifest.write_text('{}')
    with pytest.raises(ValueError, match="Legacy job manifests"):
        run_computation(["run"], None, directory=tmp_path, job=manifest)
    assert not (tmp_path / "_meta").exists()


def test_missing_remote_results_allow_inspection_to_continue(tmp_path):
    params = config("slurm")
    params["dispatch"]["share_workdir"] = True
    worker = create_worker(params, directory=tmp_path)
    worker.run([atom()])
    job = worker.job_store.get_running()[0]
    worker.scheduler.transport_name = "ssh"
    worker.scheduler.sync = lambda workdirs: None
    worker.scheduler.read_remote_file = lambda relative_path: None
    worker._sync_job(job)
    assert not worker._check_job_convergence(job)


def files_under(path):
    return {str(p.relative_to(path)): p.read_bytes() for p in path.rglob('*') if p.is_file()}


@pytest.mark.parametrize("change", ["append", "remove", "reorder", "position", "runtime", "seed", "batch", "mapping"])
def test_changed_calculation_set_leaves_metadata_and_scripts_unchanged(tmp_path, change):
    import copy

    params = config("slurm")
    frames = [atom(), atom(0.2)]
    worker = create_worker(params, directory=tmp_path)
    worker.run(frames, batch=0)
    before = files_under(tmp_path)
    params = copy.deepcopy(params)
    extra = {}
    if change == "append":
        frames.append(atom(0.3))
    elif change == "remove":
        frames.pop()
    elif change == "reorder":
        frames.reverse()
    elif change == "position":
        frames[1].positions[0, 0] += 0.01
    elif change == "runtime":
        params["executor"]["parameters"]["dump_period"] = 3
    elif change == "seed":
        extra["rng_states"] = [123, 456]
    elif change == "batch":
        params["dispatch"]["batch_size"] = 2
    restarted = create_worker(params, directory=tmp_path)
    if change == "mapping":
        restarted._make_task_plan = lambda size: [(0, 1), (0, 0)]
    with pytest.raises(ValueError, match="Calculation set conflict"):
        restarted.run(frames, batch=0, **extra)
    assert files_under(tmp_path) == before


def test_unsubmitted_batches_are_frozen_and_can_be_submitted_later(tmp_path):
    params = config("slurm")
    frames = [atom(), atom(0.2)]
    worker = create_worker(params, directory=tmp_path)
    worker.run(frames, batch=0)
    assert len(worker.metadata.inputs.read()["jobs"]) == 2
    assert len(worker.job_store) == 1
    before = worker.metadata.inputs.path.read_bytes()
    restarted = create_worker(params, directory=tmp_path)
    restarted.run(frames, batch=1)
    assert len(restarted.job_store) == 2
    assert restarted.metadata.inputs.path.read_bytes() == before
    restarted.run(frames)
    assert len(restarted.job_store) == 2


def test_omitted_seed_reuses_entire_saved_set(tmp_path):
    params = config("slurm")
    del params["executor"]["parameters"]["random_seed"]
    frames = [atom(), atom(0.2)]
    worker = create_worker(params, directory=tmp_path)
    worker.run(frames, rng_states=[123, 456], batch=0)
    before = worker.metadata.inputs.path.read_bytes()
    restarted = create_worker(params, directory=tmp_path)
    restarted.run(frames, batch=1)
    assert restarted.metadata.inputs.path.read_bytes() == before
    records = restarted.job_store.get_queued()
    assert [restarted.metadata.manifest(job.uid)["input"]["random_seeds"] for job in records] == [[123], [456]]


def test_cannot_append_worker_to_compute_plan(tmp_path):
    params = config("slurm")
    prepare_compute([params, params], [atom()], tmp_path)
    before = files_under(tmp_path)
    worker = create_worker(params, directory=tmp_path / "w2")
    worker.metadata_root = tmp_path
    with pytest.raises(ValueError, match="workers changed"):
        worker.run([atom()])
    assert files_under(tmp_path) == before


def _competing_preparation(root, position, barrier, queue):
    worker = create_worker(config("slurm"), directory=root)
    identifier, frames, batches = worker.prepare_batches([atom(position)], persist=False)
    request = worker._calculation_request(identifier, frames, batches)
    barrier.wait(timeout=20)
    try:
        worker.metadata.freeze_calculations([request])
    except ValueError as error:
        queue.put(str(error))
    else:
        queue.put("prepared")


def test_concurrent_conflicting_preparations_publish_only_one_set(tmp_path):
    context = multiprocessing.get_context("spawn")
    barrier, queue = context.Barrier(2), context.Queue()
    processes = [context.Process(target=_competing_preparation, args=(tmp_path, x, barrier, queue))
                 for x in (0.1, 0.2)]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=30)
        assert process.exitcode == 0
    outcomes = [queue.get(timeout=5) for _ in processes]
    assert outcomes.count("prepared") == 1
    assert sum("Calculation set conflict" in outcome for outcome in outcomes) == 1
    data = WorkerMetadata(tmp_path).inputs.read()
    assert len(data["structures"]) == len(data["jobs"]) == len(data["workers"]) == 1


def test_validly_hashed_job_outside_frozen_set_is_rejected(tmp_path, monkeypatch):
    import copy
    from gdpx.cli.compute import run_computation
    from gdpx.execution import factory
    from gdpx.execution.fingerprint import payload_digest

    prepare_compute(config("slurm"), [atom()], tmp_path)
    metadata = WorkerMetadata(tmp_path)
    with metadata.inputs.transaction() as data:
        saved = copy.deepcopy(next(iter(data["jobs"].values())))
        saved["input"]["wdir_names"] = ["extra"]
        saved["job_digest"] = payload_digest(saved["input"])
        data["jobs"]["extra"] = saved
    before = files_under(tmp_path)
    monkeypatch.setattr(factory, "create_worker", lambda *a, **k: pytest.fail("resolved unplanned input"))
    with pytest.raises(ValueError, match="not in the frozen calculation set"):
        run_computation(["run"], None, job="extra", directory=tmp_path)
    assert files_under(tmp_path) == before


def test_invalid_batch_does_not_create_metadata(tmp_path):
    worker = create_worker(config("slurm"), directory=tmp_path)
    with pytest.raises(ValueError, match="Unknown batch index"):
        worker.run([atom()], batch=2)
    assert not list(tmp_path.iterdir())


def test_compute_plan_cannot_adopt_existing_child_worker(tmp_path):
    params = config("slurm")
    create_worker(params, directory=tmp_path / "w0").run([atom()])
    before = files_under(tmp_path)
    with pytest.raises(ValueError, match="separate metadata"):
        prepare_compute([params, params], [atom()], tmp_path)
    assert files_under(tmp_path) == before
