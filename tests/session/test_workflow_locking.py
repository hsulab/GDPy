"""A workflow directory has only one controller, including between polls."""

import multiprocessing
from types import SimpleNamespace

import pytest

from gdpx.workflow.locking import workflow_lock


def _hold_lock(directory, ready):
    with workflow_lock(directory):
        ready.send(True)
        ready.recv()


def test_competing_process_and_crash_release(tmp_path):
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe()
    process = context.Process(target=_hold_lock, args=(tmp_path, child))
    process.start()
    try:
        assert parent.poll(20)
        assert parent.recv()
        with pytest.raises(RuntimeError, match="already running"):
            with workflow_lock(tmp_path):
                pytest.fail("competing controller acquired lock")
    finally:
        process.terminate()
        process.join(10)
        parent.close()
        child.close()
    with workflow_lock(tmp_path):
        pass


def test_exception_releases_lock(tmp_path):
    with pytest.raises(ValueError):
        with workflow_lock(tmp_path):
            raise ValueError("interrupted")
    with workflow_lock(tmp_path):
        pass


def test_api_rejects_duplicate_before_execution(tmp_path, monkeypatch):
    from gdpx.workflow.session import interface

    spec = SimpleNamespace(source=tmp_path / "active.yaml")
    monkeypatch.setattr(interface, "_run_workflow_spec", lambda *args: pytest.fail("executed"))
    with workflow_lock(tmp_path / "active"):
        with pytest.raises(RuntimeError, match="already running"):
            interface.run_workflow_spec(spec, tmp_path)


def test_cli_keeps_lock_between_polls(tmp_path, monkeypatch):
    from gdpx.cli import workflow

    spec = SimpleNamespace(source=tmp_path / "active.yaml")
    monkeypatch.setattr(workflow, "load_cli_workflow", lambda *args: spec)
    results = iter([False, True])
    def assert_locked(*args):
        with pytest.raises(RuntimeError, match="already running"):
            with workflow_lock(tmp_path / "active"):
                pass
    def execute(*args):
        assert_locked()
        return next(results)
    monkeypatch.setattr(workflow, "_run_workflow_spec", execute)
    monkeypatch.setattr(workflow.time, "sleep", assert_locked)
    assert workflow.run_workflow(spec.source, directory=tmp_path, poll_interval=1)
    with workflow_lock(tmp_path / "active"):
        pass
