import os
import signal
import time

import pytest

from gdpx.execution.processes import Process, encode_context, running_jobs, snapshot_processes, supervisor_context
from gdpx.execution.schedulers import NohupScheduler, canonicalise_scheduler


def _wait(predicate, timeout=10):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError("Timed out waiting for detached job")
        time.sleep(0.03)


@pytest.fixture
def schedulers(tmp_path):
    instances = []
    def create(name="first", commands="printf 'hello\\n'\n"):
        directory = tmp_path / (name + " with spaces")
        directory.mkdir(exist_ok=True)
        scheduler = NohupScheduler()
        scheduler.job_name = name
        scheduler.script = directory / "run.script"
        scheduler.user_commands = commands
        instances.append(scheduler)
        return scheduler
    yield create
    # Release gated test jobs even if process inspection itself fails.
    (tmp_path / "release").touch()
    scripts = {str(scheduler.script.resolve()) for scheduler in instances}
    for process in snapshot_processes():
        context = supervisor_context(process)
        if process.active and context is not None and context.get("script") in scripts:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass


def test_provider_configuration_and_rendering():
    config = {"provider": "nohup", "parameters": {"concurrent_tasks": 2,
              "environs": "export OMP_NUM_THREADS=1", "machine_prefix": "mpirun -n 2"}}
    scheduler = canonicalise_scheduler(config)
    assert isinstance(scheduler, NohupScheduler)
    assert not scheduler.is_direct
    assert scheduler.concurrent_tasks == 2
    assert "export OMP_NUM_THREADS=1" in str(scheduler)
    restored = canonicalise_scheduler(scheduler.as_dict())
    assert restored.concurrent_tasks == 2
    assert restored.machine_prefix == "mpirun -n 2"
    with pytest.raises(ValueError, match="local transport only"):
        canonicalise_scheduler(dict(config, transport={"provider": "ssh", "parameters": {
            "hostname": "cluster", "remote_wdir": "/scratch"}}))


def test_dry_run_does_not_launch_or_create_state(tmp_path):
    scheduler = NohupScheduler(is_dry_run=True)
    scheduler.script = tmp_path / "missing" / "run.script"
    assert "Attempt" in scheduler.submit(lambda: pytest.fail("executed callback"))
    assert not scheduler.script.parent.exists()


def test_detached_jobs_are_global_and_queryable_after_restart(schedulers, tmp_path, monkeypatch):
    gate = tmp_path / "release"
    commands = f'while [ ! -e "{gate}" ]; do sleep 0.05; done\nprintf "hello\\n"\n'
    first, second = schedulers(commands=commands), schedulers("second", commands)
    first_id = first.submit(lambda: pytest.fail("executed callback"))
    second_id = second.submit()
    job_ids = {first_id, second_id}
    assert len(job_ids) == 2
    assert not first.is_finished()
    with pytest.raises(RuntimeError, match="still running"):
        first.submit()
    fresh = NohupScheduler()
    fresh.script, fresh.job_name = first.script, first.job_name
    assert not fresh.is_finished()
    monkeypatch.chdir(tmp_path)
    rows = running_jobs()
    assert job_ids <= {row["job_id"] for row in rows}
    assert len([row for row in rows if row["job_id"] in job_ids]) == 2
    gate.touch()
    _wait(lambda: fresh.is_finished() and second.is_finished())
    for scheduler, job_id in ((first, first_id), (second, second_id)):
        assert "hello" in scheduler.output_path(job_id).read_text()
    assert not list(tmp_path.rglob("*.nohup"))
    assert not list(tmp_path.rglob("*.json"))
    assert first.is_finished()  # No completion record is needed for repeat queries.


def test_failed_attempt_logs_and_resubmission_are_preserved(schedulers):
    scheduler = schedulers(commands="echo failure >&2\nexit 7\n")
    first = scheduler.submit()
    _wait(scheduler.is_finished)
    first_log = scheduler.output_path(first)
    assert "status 7" in first_log.read_text()
    assert "failure" in first_log.read_text()
    scheduler.user_commands = "printf 'success\\n'\n"
    scheduler.write()
    second = scheduler.submit()
    _wait(scheduler.is_finished)
    assert second != first
    assert first_log.exists()
    assert "success" in scheduler.output_path(second).read_text()


def test_query_matches_live_supervisors_without_disk_state(schedulers, monkeypatch):
    import gdpx.execution.schedulers.nohup.nohup as module
    scheduler = schedulers()
    job_id = "nohup-00000000-0000-0000-0000-000000000001"
    monkeypatch.setattr(module, "snapshot_processes", lambda: [])
    assert scheduler.is_finished()
    monkeypatch.setattr(module, "snapshot_processes", lambda: [Process(123, 1, os.getuid(), "S", "00:01", "sleep 5")])
    assert scheduler.is_finished()
    context = encode_context(dict(job_name=scheduler.job_name, script=str(scheduler.script.resolve())))
    command = f"python /repo/_nohup_runner.py --job-id {job_id} --context {context}"
    monkeypatch.setattr(module, "snapshot_processes", lambda: [Process(123, 1, os.getuid(), "S", "00:01", command)])
    assert not scheduler.is_finished()
    monkeypatch.setattr(module, "snapshot_processes", lambda: [Process(123, 1, os.getuid(), "Z", "00:01", command)])
    assert scheduler.is_finished()
    monkeypatch.setattr(module, "snapshot_processes", lambda: [Process(123, 1, os.getuid() + 1, "S", "00:01", command)])
    assert scheduler.is_finished()
    other_script = encode_context(dict(job_name=scheduler.job_name, script="/another/run.script"))
    monkeypatch.setattr(module, "snapshot_processes", lambda: [Process(123, 1, os.getuid(), "S", "00:01",
        f"python /repo/_nohup_runner.py --job-id {job_id} --context {other_script}")])
    assert scheduler.is_finished()


def test_query_ignores_legacy_status_sidecars(schedulers):
    scheduler = schedulers()
    legacy = scheduler.script.with_name(scheduler.script.name + ".nohup")
    legacy.mkdir()
    (legacy / "current.json").write_text("{broken old status")
    assert scheduler.is_finished()
    job_id = scheduler.submit()
    _wait(scheduler.is_finished)
    assert "hello" in scheduler.output_path(job_id).read_text()
    assert (legacy / "current.json").read_text() == "{broken old status"


def test_supervisor_exec_failure_is_reported(schedulers, monkeypatch):
    import gdpx.execution.schedulers.nohup.nohup as module
    scheduler = schedulers()
    monkeypatch.setattr(module.sys, "executable", "/not-a-python-executable")
    with pytest.raises(RuntimeError, match="failed to start"):
        scheduler.submit()
    assert not list(scheduler.script.parent.glob("*.nohup"))
    assert not list(scheduler.script.parent.glob("*.json"))


def test_startup_timeout_terminates_detached_job(schedulers, monkeypatch):
    import gdpx.execution.schedulers.nohup.nohup as module
    scheduler = schedulers(commands="sleep 30\n")
    monkeypatch.setattr(module.select, "select", lambda *args: ([], [], []))
    with pytest.raises(TimeoutError, match="startup exceeded"):
        scheduler.submit()
    assert scheduler.is_finished()
    assert not list(scheduler.script.parent.glob("*.nohup"))
