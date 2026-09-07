#!/usr/bin/env python3
# -*- coding: utf-8 -*


import subprocess

from .scheduler import BaseScheduler


class PbsScheduler(BaseScheduler):

    name = "pbs"

    PREFIX = "#PBS"
    SUFFIX = ".pbs"
    SHELL = "#!/bin/bash -l"

    SUBMIT_COMMAND = "qsub"
    ENQUIRE_COMMAND = "qstat"

    default_parameters = {
        "N": "pbsJob",
        "A": None,
        "q": None,
        "l": None,
        "o": None,
        "e": None,
    }

    running_status = ["R", "Q", "PD", "CG"]

    def __str__(self):
        """Return the content of the job script."""
        content = self.SHELL + "\n"
        for key, value in self.parameters.items():
            if value:
                content += "{} -{} {}\n".format(self.PREFIX, key, value)
            # else:
            #    raise ValueError("Keyword *%s* not properly set." %key)

        content += self._convert_environs_to_content()

        if self.user_commands:
            content += "\n\n"
            content += self.user_commands

        return content

    @BaseScheduler.job_name.setter
    def job_name(self, job_name_: str):
        self._job_name = job_name_
        self.set(N=self._job_name)

    def parse_submit_output(self, output: str) -> str:
        """Return the PBS job id, retaining an optional server suffix."""
        first_line = output.strip().splitlines()[0] if output.strip() else ""
        if not first_line:
            raise RuntimeError("PBS returned empty submission output.")
        return first_line.split()[0]

    def is_finished(self) -> bool:
        proc = subprocess.Popen(
            [self.ENQUIRE_COMMAND],
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )
        stdout, _ = proc.communicate()
        return self.is_finished_from_output(stdout)

    def is_finished_from_output(self, output: str) -> bool:
        """Return whether this job name is absent from common ``qstat`` output."""
        for line in output.splitlines():
            fields = line.split()
            if self.job_name in fields:
                return False
        return True


if __name__ == "__main__":
    ...
