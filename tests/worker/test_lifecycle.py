from gdpx.execution import Runtime
from gdpx.execution.workers.drive import DriverBasedWorker
from gdpx.execution.workers.single import SingleWorker
from gdpx.providers import ComponentConfig, DispatchConfig, RuntimeConfig
from gdpx.providers.specs import PotentialSpec
from gdpx.utils.archive import ZSTD_ARCHIVE_NAME


def _runtime(driver, scheduler, *, worker="batch"):
    config = RuntimeConfig(
        potential=ComponentConfig("test"),
        executor=ComponentConfig("test", "run"),
        dispatch=DispatchConfig(worker=worker),
    )
    return Runtime(
        potential=PotentialSpec("test", {}),
        materialization=None,
        executor=driver,
        modifiers=(),
        config=config,
        scheduler=scheduler,
    )


def _create_computation_dirs(worker, tmp_path):
    for job in worker.job_store.get_running():
        for workdir in job.wdir_names:
            (tmp_path / workdir).mkdir(parents=True, exist_ok=True)


def test_runtime_worker_creates_jobs(mock_sched, fake_driver, fake_structure, tmp_path):
    worker = DriverBasedWorker(_runtime(fake_driver, mock_sched), directory=tmp_path, batchsize=2)
    worker.run([fake_structure] * 5)
    assert mock_sched.submit_count > 0
    assert len(worker.job_store.get_running()) == mock_sched.submit_count


def test_runtime_worker_marks_finished(mock_sched, fake_driver, fake_structure, tmp_path):
    worker = DriverBasedWorker(_runtime(fake_driver, mock_sched), directory=tmp_path)
    worker.run([fake_structure])
    _create_computation_dirs(worker, tmp_path)
    mock_sched.finish(True)
    worker.inspect(resubmit=False)
    assert len(worker.job_store.get_finished()) == 1


def test_runtime_worker_retrieves_and_archives(mock_sched, fake_driver, fake_structure, tmp_path):
    worker = DriverBasedWorker(_runtime(fake_driver, mock_sched), directory=tmp_path)
    worker.run([fake_structure])
    _create_computation_dirs(worker, tmp_path)
    output_path = tmp_path / "cand0" / "output.txt"
    output_path.write_text("finished")
    mock_sched.finish(True)
    assert isinstance(worker.retrieve(use_archive=True), list)
    assert (tmp_path / ZSTD_ARCHIVE_NAME).is_file()


def test_single_worker_and_conversion(mock_sched, fake_driver, fake_structure, tmp_path):
    runtime = _runtime(fake_driver, mock_sched, worker="single")
    batch_worker = DriverBasedWorker(runtime, directory=tmp_path)
    worker = SingleWorker.from_a_worker(batch_worker)
    assert isinstance(worker, DriverBasedWorker)
    worker.run([fake_structure])
    assert len(worker.job_store.get_running()) == 1


def test_runtime_config_is_the_only_serialized_worker_input(mock_sched, fake_driver, tmp_path):
    worker = DriverBasedWorker(_runtime(fake_driver, mock_sched), directory=tmp_path)
    serialized = worker.as_dict()
    assert serialized["schema_version"] == 4
    assert "potter" not in serialized
    assert "driver" not in serialized


def test_metadata_layout_resume_and_resubmit(mock_sched, fake_driver, fake_structure, tmp_path):
    worker = DriverBasedWorker(_runtime(fake_driver, mock_sched), directory=tmp_path)
    worker.run([fake_structure])
    metadata = tmp_path / "_meta"
    assert set(p.name for p in tmp_path.iterdir()) == {"_meta"}
    assert worker.job_store.path == metadata / "scheduler.json"
    assert {path.name for path in metadata.glob("*.json")} == {"inputs.json", "scheduler.json"}
    assert len(worker.metadata.inputs.read()["jobs"]) == 1
    assert len(worker.metadata.inputs.read()["structures"]) == 1
    original = worker.job_store.get_running()[0]

    restarted = DriverBasedWorker(_runtime(fake_driver, mock_sched), directory=tmp_path)
    restarted.run([fake_structure])
    assert mock_sched.submit_count == 1
    # A restarted worker reconstructs a missing script before resubmission.
    mock_sched.script.unlink()
    mock_sched.finish()
    restarted.inspect(resubmit=True)
    record = restarted.job_store.get_running()[0]
    assert record.uid == original.uid
    assert record.attempt == 2
    assert mock_sched.script.parent == metadata / "jobscripts"
    assert mock_sched.script.exists()
    assert "cd ../.. && gdp" in mock_sched.user_commands
    _create_computation_dirs(restarted, tmp_path)
    restarted.inspect()
    restarted.retrieve()
    assert len(restarted.job_store.get_retrieved()) == 1


def test_generated_driver_script_launches_from_worker_root(mock_sched, fake_driver, fake_structure, tmp_path):
    import subprocess

    from gdpx.execution.schedulers.slurm import SlurmScheduler
    mock_sched = SlurmScheduler(is_dry_run=True)
    root = tmp_path / "run with spaces"
    mock_sched.environs = 'gdp() { pwd > launch-cwd; printf "%s\\n" "$@" > launch-args; }'
    worker = DriverBasedWorker(_runtime(fake_driver, mock_sched), directory=root)
    worker.run([fake_structure])
    subprocess.run(["bash", str(mock_sched.script)], cwd=mock_sched.script.parent, check=True)
    assert (root / "launch-cwd").read_text().strip() == str(root)
    args = (root / "launch-args").read_text().splitlines()
    assert args[:3] == ["compute", "run", "--job"]
    assert args[3] in worker.metadata.inputs.read()["jobs"]


def test_legacy_driver_layout_is_not_overwritten(mock_sched, fake_driver, tmp_path):
    import pytest

    legacy = tmp_path / "_mock_jobs.json"
    legacy.write_text('{"_default": {}}')
    worker = DriverBasedWorker(_runtime(fake_driver, mock_sched), directory=tmp_path)
    with pytest.raises(RuntimeError, match="Legacy driver worker layout"):
        worker.inspect()
    assert legacy.read_text() == '{"_default": {}}'
    assert not (tmp_path / "_meta").exists()


def test_single_rewind_uses_metadata_store(mock_sched, fake_driver, fake_structure, tmp_path):
    worker = SingleWorker(_runtime(fake_driver, mock_sched), directory=tmp_path)
    worker.wdir_name = "cand2"
    worker.run([fake_structure])
    assert len(worker.job_store) == 1
    worker.rewind_to_step(1)
    assert len(worker.job_store) == 0
    assert not (tmp_path / "_mock_jobs.json").exists()


def test_scheduler_change_rejected(mock_sched, fake_driver, fake_structure, tmp_path):
    import pytest
    from gdpx.execution.schedulers.direct import DirectScheduler

    worker = DriverBasedWorker(_runtime(fake_driver, mock_sched), directory=tmp_path)
    worker.run([fake_structure])
    before = worker.job_store.path.read_bytes()
    worker.scheduler = DirectScheduler()
    with pytest.raises(ValueError, match="provider changed"):
        worker.inspect()
    assert (tmp_path / "_meta" / "scheduler.json").read_bytes() == before


def test_remote_driver_sync_preserves_jobs_and_updates_cache(tmp_path):
    import shutil
    from pathlib import Path
    from types import SimpleNamespace
    from gdpx.execution.schedulers.direct import DirectScheduler
    from gdpx.execution.schedulers.remote import SshTransport

    local = tmp_path / "local"
    remote = tmp_path / "remote" / "job"
    for root in (local, remote):
        (root / "_meta").mkdir(parents=True)
        (root / "cand0").mkdir()
    (local / "_meta" / "_scheduler.json").write_text("local jobs")
    (remote / "_meta" / "_scheduler.json").write_text("stale remote jobs")
    (remote / "_meta" / "test_cache.xyz").write_text("cached results")
    (local / "cand0" / "obsolete").write_text("old")
    (remote / "cand0" / "result").write_text("new")

    class Sftp:
        def listdir_attr(self, directory):
            return [SimpleNamespace(filename=p.name, st_mode=p.stat().st_mode)
                    for p in Path(directory).iterdir()]
        def lstat(self, path):
            return Path(path).stat()
        stat = lstat
        def get(self, source, destination):
            shutil.copyfile(source, destination)
        def close(self):
            pass

    transport = SshTransport(DirectScheduler(), "test", str(remote.parent))
    transport.local_root = local
    transport.script = local / "_meta" / "run.script"
    transport.job_name = "job"
    transport._client = lambda: SimpleNamespace(open_sftp=lambda: Sftp(), close=lambda: None)
    transport.sync(["cand0"])
    assert (local / "_meta" / "_scheduler.json").read_text() == "local jobs"
    assert (local / "_meta" / "test_cache.xyz").read_text() == "cached results"
    assert (local / "cand0" / "result").read_text() == "new"
    assert not (local / "cand0" / "obsolete").exists()


def test_runtime_and_seed_changes_do_not_reuse_jobs(mock_sched, fake_driver, fake_structure, tmp_path):
    import pytest

    worker = DriverBasedWorker(_runtime(fake_driver, mock_sched), directory=tmp_path)
    worker.run([fake_structure], rng_states=[123])
    record = worker.job_store.get_running()[0]
    assert len(record.structure_digest) == 64
    assert len(record.job_digest) == 64
    assert not record.md5
    before = worker.job_store.path.read_bytes()
    with pytest.raises(ValueError, match="Calculation set conflict"):
        worker.run([fake_structure], rng_states=[124])
    worker._share_wdir = True
    with pytest.raises(ValueError, match="Calculation set conflict"):
        worker.run([fake_structure], rng_states=[123])
    assert worker.job_store.path.read_bytes() == before
    assert mock_sched.submit_count == 1


def test_resubmit_uses_exact_saved_batch_and_rng_state(mock_sched, fake_driver, fake_structure, tmp_path):
    import numpy as np

    state = np.random.default_rng(765).bit_generator.state
    worker = SingleWorker(_runtime(fake_driver, mock_sched, worker="single"), directory=tmp_path)
    worker.wdir_name = "cand7"
    worker.run([fake_structure], rng_states=[state])
    restarted = SingleWorker(_runtime(fake_driver, mock_sched, worker="single"), directory=tmp_path)
    mock_sched.finish()
    restarted.inspect(resubmit=True)
    arguments = mock_sched.submitted_jobs[-1]["func"].keywords
    assert arguments["rng_states"] == [state]
    assert arguments["computation_dirnames"] == ["cand7"]
    assert mock_sched.submit_count == 2


def test_same_structure_in_different_single_workdirs_is_rejected(
        mock_sched, fake_driver, fake_structure, tmp_path):
    import pytest
    worker = SingleWorker(_runtime(fake_driver, mock_sched), directory=tmp_path)
    worker.wdir_name = "cand0"
    worker.run([fake_structure])
    before = worker.metadata.inputs.path.read_bytes()
    worker.wdir_name = "cand1"
    with pytest.raises(ValueError, match="wdir_names changed"):
        worker.run([fake_structure])
    assert worker.metadata.inputs.path.read_bytes() == before
    assert len(worker.job_store.get_running()) == 1


def test_legacy_md5_records_rejected_without_modification(mock_sched, fake_driver, tmp_path):
    import json
    import pytest

    metadata = tmp_path / "_meta"
    metadata.mkdir()
    path = metadata / "_scheduler.json"
    content = json.dumps({"_default": {"1": {"md5": "a" * 32, "queued": True}}})
    path.write_text(content)
    worker = DriverBasedWorker(_runtime(fake_driver, mock_sched), directory=tmp_path)
    with pytest.raises(ValueError, match="Legacy driver metadata"):
        worker.inspect()
    assert path.read_text() == content
