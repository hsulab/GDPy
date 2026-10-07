"""Detached local jobs queried through the host process table."""

import os
import pathlib
import select
import subprocess
import sys
import threading
import uuid

from gdpx.execution.processes import encode_context, snapshot_processes, supervisor_context

from ..scheduler import BaseScheduler


class NohupScheduler(BaseScheduler):
    name = "nohup"
    supports_concurrent_tasks = True
    SHELL = "#!/bin/bash -l"

    @BaseScheduler.job_name.setter
    def job_name(self, value):
        self._job_name = value

    def output_path(self, job_id):
        """Return the ordinary scheduler output file for one attempt."""
        return self.script.resolve().parent / f"{job_id}.out"

    def __str__(self):
        return self.SHELL + self._convert_environs_to_content() + self.user_commands + "\n"

    def is_finished(self):
        for process in snapshot_processes():
            if not process.active or process.uid != os.getuid():
                continue
            context = supervisor_context(process)
            if (context is not None and context.get("job_name") == self.job_name
                    and context.get("script") == str(self.script.resolve())):
                return False
        return True

    def submit(self, func_to_execute=None):
        if self.is_dry_run:
            return f"Attempt to submit {self.script.name} with nohup."
        if os.name != "posix":
            raise RuntimeError("The nohup scheduler requires a POSIX host.")
        if not self.is_finished():
            raise RuntimeError(f"Nohup job for {self.script} is still running.")
        if not self.script.exists():
            self.write()
        job_id = "nohup-" + str(uuid.uuid4())
        script = self.script.resolve()
        directory = script.parent
        if directory.name == "jobscripts" and directory.parent.name == "_meta":
            directory = directory.parent.parent
        elif directory.name == "_meta":
            directory = directory.parent
        context = dict(script=str(script), directory=str(directory), job_name=self.job_name)
        runner = pathlib.Path(__file__).with_name("_nohup_runner.py")
        ready_read, ready_write = os.pipe()
        process = None
        try:
            context["ready_fd"] = ready_write
            with open(self.output_path(job_id), "wb") as output:
                process = subprocess.Popen(
                    ["nohup", sys.executable, str(runner), "--job-id", job_id,
                     "--context", encode_context(context)],
                    cwd=script.parent, stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT,
                    start_new_session=True, pass_fds=(ready_write,),
                )
            os.close(ready_write)
            ready_write = None
            # Startup acknowledgement is transient IPC, not scheduler state on disk.
            readable, _, _ = select.select([ready_read], [], [], max(0, self.submit_timeout))
            if not readable:
                raise TimeoutError(f"Nohup supervisor startup exceeded {self.submit_timeout}s")
            if os.read(ready_read, 1) != b"R":
                process.wait()
                raise RuntimeError(f"Nohup supervisor failed to start; see {self.output_path(job_id)}")
        except BaseException:
            if process is not None:
                if process.poll() is None:
                    try:
                        os.killpg(process.pid, 15)
                    except ProcessLookupError:
                        pass
                process.wait()
            raise
        finally:
            os.close(ready_read)
            if ready_write is not None:
                os.close(ready_write)
        # Reap children if the submitting process stays alive; jobs also survive
        # its exit because this waiting thread is a daemon.
        threading.Thread(target=process.wait, daemon=True).start()
        return job_id
