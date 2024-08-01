#!/usr/bin/env python3
# -*- coding: utf-8 -*


import re
import subprocess

from .scheduler import AbstractScheduler


class SlurmScheduler(AbstractScheduler):
    """Slurm scheduler.

    A slurm scheduler.

    """

    name = "slurm"

    PREFIX = "#SBATCH"
    SUFFIX = ".slurm"
    SHELL = "#!/bin/bash -l"

    SUBMIT_COMMAND = "sbatch"
    ENQUIRE_COMMAND = '`which squeue` -u `whoami` --format="%.12i %.12P %.60j %.4t %.12M %.12L %.5D %.4C"'

    # compability for different machine
    # not all keywords are necessary
    default_parameters = {
        "job-name": "slurmJob",
        "account": None,
        "partition": None,
        "time": None,
        # - CPU
        "nodes": None,
        "ntasks": None,
        "tasks-per-node": None,
        "cpus-per-task": None,
        "mem-per-cpu": None,  # "4G"
        # "output": "slurm.o%j",
        # "error": "slurm.e%j"
        # - GPU
        "gres": None,
        "mem-per-gpu": None,  # "32G"
    }

    running_status = ["R", "Q", "PD", "CG"]

    def __str__(self) -> str:
        """Return the content of the job script."""
        # - slurm params
        content = self.SHELL + "\n"
        for key, value in self.parameters.items():
            if value:
                content += "{} --{}={}\n".format(self.PREFIX, key, value)
            # else:
            #    raise ValueError("Keyword *%s* not properly set." %key)

        if self.environs:
            content += "\n\n"
            content += self.environs

        if self.user_commands:
            content += "\n\n"
            content += self.user_commands

        return content

    @AbstractScheduler.job_name.setter
    def job_name(self, job_name_: str):
        self._job_name = job_name_
        self.set(**{"job-name": self._job_name})
        return

    def is_finished(self) -> bool:
        """Check if the job were finished.

        Currently, we only check whether the job name has been found in the queue.
        If found, we assume the job is still running. Otherwise, it may have already
        finished.

        Returns:
            Whether the jobs is finished. True for finished, False otherwise.

        """
        # - run enquire
        p = subprocess.Popen(
            [self.ENQUIRE_COMMAND],
            shell=True,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            close_fds=True,
            universal_newlines=True,
        )
        output = p.stdout
        lines = output.readlines()
        content = "".join(lines)

        pattern = re.compile(
            r"\s+(\d+)\s+\S+\s+(\S+)\s+[A-Z]+\s+\S+\s+\S+\s+\d+\s+\d+"
        )
        matches = pattern.findall(content)
        names = [m[1] for m in matches]
        if self.job_name not in names:
            finished = True
        else:
            finished = False

        return finished


if __name__ == "__main__":
    ...
