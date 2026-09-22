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
                options={})


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
    params["options"]["retain_info"] = True
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
    params["options"]["share_workdir"] = True
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


@pytest.mark.parametrize("mode", ["worker", "plan"])
def test_previous_sha256_layout_resumes_without_migration(tmp_path, mode):
    # This discriminator selects the supported pre-consolidation storage adapter.
    metadata = tmp_path / "_meta"
    metadata.mkdir()
    (metadata / "_scheduler.json").write_text('{"_default": {}}')
    if mode == "worker":
        worker = create_worker(config(), directory=tmp_path)
        worker.run([atom()])
        worker.inspect()
        state = worker.job_store.path.read_bytes()
        restarted = create_worker(config(), directory=tmp_path)
        restarted.run([atom()])
        assert restarted.job_store.path.read_bytes() == state
        assert len(restarted.retrieve()) == 1
        assert list(metadata.glob("job-*.json"))
        assert list(metadata.glob("*.atoms.json"))
    else:
        plan = prepare_compute(config(), [atom()], tmp_path)
        assert plan.schema_version == 4
        assert plan.path.name == "compute-plan.json"
        submit_compute(plan)
        restarted = load_compute_plan(tmp_path)
        assert submit_compute(restarted).submitted_batches == ()
        assert collect_compute(restarted).number_of_trajectories == 1
    assert not (metadata / "inputs.json").exists()
    assert not (metadata / "scheduler.json").exists()


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


def test_legacy_manifest_argument_still_executes(tmp_path):
    from gdpx.cli.compute import run_computation

    (tmp_path / "_meta").mkdir()
    (tmp_path / "_meta" / "_scheduler.json").write_text('{"_default": {}}')
    worker = create_worker(config("slurm"), directory=tmp_path)
    worker.run([atom()])
    manifest = next((tmp_path / "_meta").glob("job-*.json"))
    before = worker.job_store.path.read_bytes()
    run_computation(["run"], None, directory=tmp_path, job=manifest)
    assert (tmp_path / "cand0").exists()
    assert worker.job_store.path.read_bytes() == before
    assert not (tmp_path / "_meta" / "inputs.json").exists()


def test_missing_remote_results_allow_inspection_to_continue(tmp_path):
    params = config("slurm")
    params["options"]["share_workdir"] = True
    worker = create_worker(params, directory=tmp_path)
    worker.run([atom()])
    job = worker.job_store.get_running()[0]
    worker.scheduler.transport_name = "ssh"
    worker.scheduler.sync = lambda workdirs: None
    worker.scheduler.read_remote_file = lambda relative_path: None
    worker._sync_job(job)
    assert not worker._check_job_convergence(job)
