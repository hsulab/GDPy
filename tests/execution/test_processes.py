import json
import os

import pytest

from gdpx.execution import processes as module
from gdpx.execution.processes import Process, encode_context, parse_processes, running_jobs


def _process(pid, command, *, parent=1, uid=None, state="S"):
    return Process(pid, parent, os.getuid() if uid is None else uid, state, "01:02", command)


def test_ps_parsing_preserves_command_and_elapsed():
    output = """
  123 1 501 Ss 2-03:04:05 /opt/env/bin/python /opt/env/bin/gdp -d results compute run --job abc
  124 123 501 Z 00:01 [python] <defunct>
bad output
"""
    processes = parse_processes(output)
    assert len(processes) == 2
    assert processes[0].elapsed == "2-03:04:05"
    assert processes[0].command.endswith("compute run --job abc")
    assert processes[0].active
    assert not processes[1].active


@pytest.mark.parametrize("command", [
    "/env/bin/gdp -d results compute input.xyz",
    "/env/bin/python /env/bin/gdp --directory=results compute run --job abc",
    "python3.12 -m gdpx.main compute run --job abc",
    "gdp -r compute -d status explore config.yaml",
    "gdp workflow run workflow.yaml",
    "gdp compute --batch 0 input.xyz",
    "gdp -r runtime.yaml compute",
    "gdp --log= compute run --job abc",
])
def test_identifies_simulation_launchers(command, monkeypatch):
    monkeypatch.setattr(module, "process_directory", lambda pid: "/work")
    assert running_jobs(processes=[_process(123, command)])[0]["job_id"] == "pid:123"


@pytest.mark.parametrize("command", [
    "gdp compute prepare input.xyz", "gdp compute submit", "gdp compute status",
    "gdp compute collect", "gdp compute resubmit", "gdp queue --json",
    "gdp compute --plan inputs.json status", "gdp compute --batch 0 resubmit",
    "gdp compute --help", "gdp explore --help",
    "gdp workflow status workflow.yaml", "gdp workflow validate workflow.yaml",
    "gdp build config.yaml", "bash -c 'gdp compute run --job abc'",
    "python -c 'print(\"gdp compute run\")'", "grep 'gdp compute'",
    "other-gdp compute input.xyz",
])
def test_excludes_non_simulation_processes(command):
    assert running_jobs(processes=[_process(123, command)]) == []


def test_global_listing_filters_users_states_and_descendants(monkeypatch):
    monkeypatch.setattr(module, "process_directory", lambda pid: f"/work/{pid}")
    context = encode_context({"directory": "/unrelated directory"})
    jobs = [
        _process(30, f"/env/bin/python /repo/_nohup_runner.py --job-id nohup-abc --context {context}"),
        _process(31, "bash run.script", parent=30),
        _process(32, "gdp compute run --job abc", parent=31),
        _process(33, "gdp compute run --job abc --task 1", parent=32),
        _process(10, "gdp compute input.xyz", state="T"),
        _process(11, "gdp compute input.xyz", parent=10),
        _process(20, "gdp explore config.yaml", uid=os.getuid() + 10000),
        _process(40, "gdp compute input.xyz", state="Z"),
    ]
    rows = running_jobs(processes=jobs)
    assert [row["pid"] for row in rows] == [10, 30]
    assert rows[1]["job_id"] == "nohup-abc"
    assert rows[1]["directory"] == "/unrelated directory"
    assert [row["pid"] for row in running_jobs(processes=jobs, all_users=True)] == [10, 20, 30]


def test_supervisor_paths_with_spaces(monkeypatch):
    monkeypatch.setattr(module, "process_directory", lambda pid: pytest.fail("looked up managed directory"))
    context = encode_context({"directory": "/work with spaces", "script": "/work with spaces/run.script"})
    process = _process(123, f"/env with spaces/bin/python /repo with spaces/_nohup_runner.py "
                       f"--job-id nohup-abc --context {context}")
    row = running_jobs(processes=[process])[0]
    assert row["job_id"] == "nohup-abc"
    assert row["directory"] == "/work with spaces"
    assert row["command"] == "bash -l '/work with spaces/run.script'"


@pytest.mark.parametrize("platform", ["linux", "darwin"])
def test_directory_lookup(platform, monkeypatch):
    monkeypatch.setattr(module.sys, "platform", platform)
    if platform == "linux":
        monkeypatch.setattr(module.os, "readlink", lambda path: "/work with spaces")
    else:
        from types import SimpleNamespace
        monkeypatch.setattr(module.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(
            stdout="p123\nfcwd\nn/work with spaces\n"))
    assert module.process_directory(123) == "/work with spaces"


def test_directory_permission_error_is_unavailable(monkeypatch):
    monkeypatch.setattr(module.sys, "platform", "linux")
    def denied(path):
        raise PermissionError(path)
    monkeypatch.setattr(module.os, "readlink", denied)
    assert module.process_directory(123) is None


def test_queue_cli_is_read_only_and_clean_json(tmp_path, monkeypatch, capsys):
    import sys
    from gdpx import main
    from gdpx.cli import queue

    monkeypatch.setattr(queue, "running_jobs", lambda **kwargs: [])
    monkeypatch.setattr(main, "bootstrap_registries", lambda **kwargs: pytest.fail("bootstrapped registry"))
    target = tmp_path / "must-not-exist"
    monkeypatch.setattr(sys, "argv", ["gdp", "-d", str(target), "queue", "--json", "--all-users"])
    main.main()
    assert json.loads(capsys.readouterr().out) == []
    assert not target.exists()


def test_queue_table_and_filter_arguments(monkeypatch, capsys):
    from gdpx.cli import queue

    calls = []
    def jobs(**kwargs):
        calls.append(kwargs)
        return [dict(job_id="pid:123", pid=123, user="alice", state="T", elapsed="00:01",
                     directory=None, command="gdp compute input.xyz")]
    monkeypatch.setattr(queue, "running_jobs", jobs)
    queue.run_queue(all_users=True)
    output = capsys.readouterr().out
    assert "JOBID" in output and "pid:123" in output and "—" in output
    assert calls == [{"all_users": True}]
