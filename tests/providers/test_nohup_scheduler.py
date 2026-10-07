import json
import os
import pathlib
import signal
import time

import pytest

from gdpx.execution.processes import Process, running_jobs
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
    for scheduler in instances:
        current = scheduler.state_directory / "current.json"
        if current.exists():
            record = json.loads(current.read_text())
            # PID reuse is checked by production discovery before killing test jobs.
            try:
                if not scheduler.is_finished():
                    os.killpg(record["pid"], signal.SIGTERM)
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
    job_ids = {first.submit(lambda: pytest.fail("executed callback")), second.submit()}
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
    for scheduler in (first, second):
        record = scheduler._current()
        attempt = scheduler.state_directory / record["job_id"]
        assert "hello" in (attempt / "output.log").read_text()
        assert json.loads((attempt / "completed.json").read_text())["exit_code"] == 0


def test_failed_attempt_logs_and_resubmission_are_preserved(schedulers):
    scheduler = schedulers(commands="echo failure >&2\nexit 7\n")
    first = scheduler.submit()
    _wait(scheduler.is_finished)
    first_attempt = scheduler.state_directory / first
    assert json.loads((first_attempt / "completed.json").read_text())["exit_code"] == 7
    assert "failure" in (first_attempt / "output.log").read_text()
    scheduler.user_commands = "printf 'success\\n'\n"
    scheduler.write()
    second = scheduler.submit()
    _wait(scheduler.is_finished)
    assert second != first
    assert first_attempt.exists()
    assert "success" in (scheduler.state_directory / second / "output.log").read_text()


def test_query_checks_pid_identity_and_missing_process(schedulers, monkeypatch):
    import gdpx.execution.schedulers.nohup.nohup as module
    scheduler = schedulers()
    scheduler.state_directory.mkdir()
    job_id = "nohup-00000000-0000-0000-0000-000000000001"
    current = scheduler.state_directory / "current.json"
    current.write_text(json.dumps(dict(job_id=job_id, pid=123, job_name=scheduler.job_name)))
    monkeypatch.setattr(module, "snapshot_processes", lambda: [])
    assert scheduler.is_finished()
    monkeypatch.setattr(module, "snapshot_processes", lambda: [Process(123, 1, os.getuid(), "S", "00:01", "sleep 5")])
    assert scheduler.is_finished()
    current.write_text("{broken")
    with pytest.raises(RuntimeError, match="corrupt nohup state"):
        scheduler.is_finished()
    current.unlink()
    with pytest.raises(RuntimeError, match="Missing"):
        scheduler.is_finished()


def test_supervisor_exec_failure_is_reported(schedulers, monkeypatch):
    import gdpx.execution.schedulers.nohup.nohup as module
    scheduler = schedulers()
    monkeypatch.setattr(module.sys, "executable", "/not-a-python-executable")
    with pytest.raises(RuntimeError, match="failed to start"):
        scheduler.submit()
    assert not (scheduler.state_directory / "current.json").exists()
