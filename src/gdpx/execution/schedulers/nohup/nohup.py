"""Detached local jobs with persistent status and attempt logs."""

import json
import os
import pathlib
import subprocess
import sys
import time
import uuid

from gdpx.execution.processes import encode_context, snapshot_processes, supervisor_context

from ._nohup_runner import atomic_json
from ..scheduler import BaseScheduler


class NohupScheduler(BaseScheduler):
    name = "nohup"
    supports_concurrent_tasks = True
    SHELL = "#!/bin/bash -l"

    @BaseScheduler.job_name.setter
    def job_name(self, value):
        self._job_name = value

    @property
    def state_directory(self):
        return self.script.resolve().with_name(self.script.name + ".nohup")

    def __str__(self):
        return self.SHELL + self._convert_environs_to_content() + self.user_commands + "\n"

    def _current(self):
        try:
            record = json.loads((self.state_directory / "current.json").read_text())
            if (not isinstance(record, dict) or not isinstance(record["pid"], int)
                    or record["pid"] <= 0 or not record["job_id"].startswith("nohup-")
                    or record["job_name"] != self.job_name):
                raise ValueError("invalid job identity")
            # Construct the attempt path from the ID, never from untrusted stored paths.
            uuid.UUID(record["job_id"].removeprefix("nohup-"))
            return record
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
            raise RuntimeError(f"Missing or corrupt nohup state for {self.script}: {error}") from error

    def is_finished(self):
        record = self._current()
        completed = self.state_directory / record["job_id"] / "completed.json"
        if completed.exists():
            try:
                result = json.loads(completed.read_text())
                if result["job_id"] != record["job_id"] or not isinstance(result["exit_code"], int):
                    raise ValueError("invalid completion record")
            except (OSError, ValueError, KeyError, TypeError) as error:
                raise RuntimeError(f"Corrupt nohup completion record: {completed}") from error
            return True
        for process in snapshot_processes():
            if process.pid == record["pid"] and process.active:
                context = supervisor_context(process)
                return context is None or context["job_id"] != record["job_id"]
        return True

    def submit(self, func_to_execute=None):
        if self.is_dry_run:
            return f"Attempt to submit {self.script.name} with nohup."
        if os.name != "posix":
            raise RuntimeError("The nohup scheduler requires a POSIX host.")
        if (self.state_directory / "current.json").exists() and not self.is_finished():
            raise RuntimeError(f"Nohup job for {self.script} is still running.")
        if not self.script.exists():
            self.write()
        job_id = "nohup-" + str(uuid.uuid4())
        attempt = self.state_directory / job_id
        attempt.mkdir(parents=True)
        script = self.script.resolve()
        directory = script.parent
        if directory.name == "jobscripts" and directory.parent.name == "_meta":
            directory = directory.parent.parent
        elif directory.name == "_meta":
            directory = directory.parent
        context = dict(script=str(script), directory=str(directory),
                       attempt_directory=str(attempt), job_name=self.job_name)
        runner = pathlib.Path(__file__).with_name("_nohup_runner.py")
        with open(attempt / "output.log", "wb") as output:
            process = subprocess.Popen(
                ["nohup", sys.executable, str(runner), "--job-id", job_id,
                 "--context", encode_context(context)],
                cwd=script.parent, stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        record = dict(context, job_id=job_id, pid=process.pid, submitted_at=time.time())
        # Wait only for supervisor startup, so a failed exec is not reported as a submission.
        deadline = time.monotonic() + self.submit_timeout
        while not (attempt / "started.json").exists():
            if process.poll() is not None:
                raise RuntimeError(f"Nohup supervisor failed to start; see {attempt / 'output.log'}")
            if time.monotonic() >= deadline:
                try:
                    os.killpg(process.pid, 15)
                except ProcessLookupError:
                    pass
                process.wait()
                raise TimeoutError(f"Nohup supervisor startup exceeded {self.submit_timeout}s")
            time.sleep(0.01)
        try:
            atomic_json(self.state_directory / "current.json", record)
        except BaseException:
            try:
                os.killpg(process.pid, 15)
            except ProcessLookupError:
                pass
            process.wait()
            raise
        return job_id
