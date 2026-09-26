"""Slurm scheduler integration."""

import subprocess
from typing import ClassVar

from .scheduler import BaseScheduler


class SlurmScheduler(BaseScheduler):
    """Submit jobs with Slurm and inspect their queue state."""

    name = "slurm"
    supports_concurrent_tasks = True

    PREFIX = "#SBATCH"
    SUFFIX = ".slurm"
    SHELL = "#!/bin/bash -l"

    SUBMIT_COMMAND = "sbatch"
    ENQUIRE_COMMAND = 'squeue -u "$(whoami)" --format="%.12i %.12P %.60j %.4t %.12M %.12L %.5D %.4C"'

    default_parameters: ClassVar[dict[str, object]] = {
        "job-name": "slurmJob",
        "account": None,
        "partition": None,
        "time": None,
        "nodes": None,
        "ntasks": None,
        "tasks-per-node": None,
        "cpus-per-task": None,
        "mem-per-cpu": None,
        "gres": None,
        "mem-per-gpu": None,
    }

    running_status: ClassVar[list[str]] = ["R", "Q", "PD", "CG"]

    def __str__(self) -> str:
        content = self.SHELL + "\n"
        for key, value in self.parameters.items():
            if value is not None:
                content += f"{self.PREFIX} --{key}={value}\n"
        content += self._convert_environs_to_content()
        if self.user_commands:
            content += "\n\n" + self.user_commands
        return content

    @BaseScheduler.job_name.setter
    def job_name(self, job_name_: str):
        self._job_name = job_name_
        self.set(**{"job-name": self._job_name})

    def is_finished(self) -> bool:
        proc = subprocess.Popen(
            self.ENQUIRE_COMMAND,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )
        stdout, _ = proc.communicate()
        return self.is_finished_from_output(stdout)

    def is_finished_from_output(self, output: str) -> bool:
        """Return whether this job name is absent from ``squeue`` output."""
        for line in output.splitlines():
            fields = line.split()
            if len(fields) >= 3 and fields[2] == self.job_name:
                return False
        return True
