import json
import os
import pathlib
import shlex
import shutil
import signal
import subprocess
import sys
import time

import pytest
import yaml
from ase import Atoms
from ase.io import read, write

from gdpx.execution.lifecycle import inspect_compute, load_compute_plan
from gdpx.execution.processes import snapshot_processes, supervisor_context


def _wait(predicate, timeout=30):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError("Timed out waiting for compute job")
        time.sleep(0.1)


@pytest.fixture
def calculation(tmp_path):
    directory = tmp_path / "results with spaces"
    inputs = tmp_path / "structures.xyz"
    atoms = Atoms("Cu", positions=[[0, 0, 0]], cell=[4, 4, 4], pbc=True)
    second = atoms.copy()
    second.positions[0, 0] = 0.1
    write(inputs, [atoms, second])
    runtime = tmp_path / "runtime.yaml"
    environ = f"export PATH={shlex.quote(str(pathlib.Path(sys.executable).parent))}:$PATH\n"
    config = {
        "schema_version": 4, "potential": {"provider": "emt", "parameters": {}},
        "executor": {"provider": "ase", "method": "min", "parameters": {
            "steps": 1, "fmax": 0.5, "dump_period": 1, "random_seed": 17}},
        "scheduler": {"provider": "nohup", "parameters": {"environs": environ}},
        "dispatch": {"batch_size": 1},
    }
    def cli(*args, check=True, log=""):
        return subprocess.run(
            [sys.executable, shutil.which("gdp"), "--log", log, "-d", str(directory), *map(str, args)],
            cwd=tmp_path, capture_output=True, text=True, check=check, timeout=30,
        )
    yield directory, inputs, runtime, config, cli
    for process in snapshot_processes():
        context = supervisor_context(process)
        if (process.active and context is not None and context.get("script")
                and directory in pathlib.Path(context["script"]).parents):
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass


def test_nohup_cli_submit_query_retrieve_across_processes(calculation, tmp_path):
    directory, inputs, runtime, config, cli = calculation
    gate = tmp_path / "release"
    config["scheduler"]["parameters"]["environs"] += f'while [ ! -e "{gate}" ]; do sleep 0.05; done\n'
    runtime.write_text(yaml.safe_dump(config))
    cli("-r", runtime, "compute", "prepare", inputs)
    assert not list(directory.glob("_meta/jobscripts/*.nohup"))
    cli("compute", "submit", log="gdp.out")
    shared_log = directory / "gdp.out"
    submission_log = shared_log.read_bytes()
    records = json.loads((directory / "_meta/scheduler.json").read_text())["_default"]
    assert len(records) == 2  # Queued dispatch honors batch_size instead of direct batching.
    job_ids = {record["scheduler_job_id"] for record in records.values()}
    assert not list(directory.glob("_meta/jobscripts/*.nohup"))
    assert not list(directory.glob("_meta/jobscripts/**/*.json"))
    before = (directory / "_meta/scheduler.json").read_bytes()
    cli("compute", "submit")
    assert (directory / "_meta/scheduler.json").read_bytes() == before
    status = cli("compute", "status")
    assert "running" in status.stdout + status.stderr
    rows = json.loads(cli("queue", "--json").stdout)
    assert job_ids <= {row["job_id"] for row in rows}
    assert cli("compute", "collect", check=False).returncode != 0
    gate.touch()
    _wait(lambda: inspect_compute(load_compute_plan(directory)).state == "finished")
    assert shared_log.read_bytes() == submission_log
    for job_id in job_ids:
        output = directory / "_meta/jobscripts" / f"{job_id}.out"
        assert "____" in output.read_text()  # Child CLI diagnostics remain captured.
    status = cli("compute", "status")
    assert "finished" in status.stdout + status.stderr
    cli("compute", "collect")
    assert len(read(directory / "results/end_frames.xyz", ":")) == 2
    state = json.loads((directory / "_meta/scheduler.json").read_text())
    assert {record["scheduler_job_id"] for record in state["_default"].values()} == job_ids
    assert {record["attempt"] for record in state["_default"].values()} == {1}


def test_nohup_failed_calculation_requires_explicit_resubmission(calculation, tmp_path):
    directory, inputs, runtime, config, cli = calculation
    allow = tmp_path / "allow"
    config["scheduler"]["parameters"]["environs"] += f'test -e "{allow}" || exit 7\n'
    config["dispatch"]["batch_size"] = 2
    runtime.write_text(yaml.safe_dump(config))
    cli("-r", runtime, "compute", "prepare", inputs)
    cli("compute", "submit")
    state_path = directory / "_meta/scheduler.json"
    first = next(iter(json.loads(state_path.read_text())["_default"].values()))["scheduler_job_id"]
    first_log = directory / "_meta/jobscripts" / f"{first}.out"
    _wait(lambda: first_log.exists() and "status 7" in first_log.read_text())
    assert inspect_compute(directory).state == "running"
    assert cli("compute", "collect", check=False).returncode != 0
    allow.touch()
    cli("compute", "resubmit", "--batch", "0")
    second = next(iter(json.loads(state_path.read_text())["_default"].values()))["scheduler_job_id"]
    assert second != first
    _wait(lambda: inspect_compute(directory).state == "finished")
    cli("compute", "collect")
    assert len(read(directory / "results/end_frames.xyz", ":")) == 2
    record = next(iter(json.loads((directory / "_meta/scheduler.json").read_text())["_default"].values()))
    assert record["attempt"] == 2
    assert first_log.exists()
    assert not list(directory.glob("_meta/jobscripts/*.nohup"))
