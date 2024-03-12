#!/usr/bin/env python3
# -*- coding: utf-8 -*

import subprocess

from .scheduler import AbstractScheduler


class LsfScheduler(AbstractScheduler):
    """Load Sharing Facility (LSF) scheduler.

    A LSF scheduler. Commands are bjobs, bsub, bkill.

    """

    name = "lsf"

    PREFIX = "#BSUB"
    SUFFIX = ".lsf"
    SHELL = "#!/bin/bash -l"

    SUBMIT_COMMAND = "bsub < "
    # ENQUIRE_COMMAND = "`which squeue` -u `whoami` --format=\"%.12i %.12P %.60j %.4t %.12M %.12L %.5D %.4C\""
    ENQUIRE_COMMAND = "`which bjobs` -u `whoami` -w"

    # compability for different machine
    # not all keywords are necessary
    default_parameters = {
        "J": "lsfJob",  # job name
        "q": None,  # partition
        "W": None,  # time
        # - CPU
        "R": None,  # nodes
        "n": None,  # ncpus
        "M": None,  # memory
        "o": "lsf.o%J",  # output
        "e": "lsf.e%J",  # error
        # - GPU
        # "gres": None,
        # "mem-per-gpu": None, # "32G"
    }

    running_status = ["RUN", "PEND", "DONE", "EXIT", "SSUSP", "USUSP"]

    def __str__(self) -> str:
        """Return the content of the job script."""
        # - scheduler params
        content = self.SHELL + "\n"
        for key, value in self.parameters.items():
            if value:
                # content += "{} --{}={}\n".format(self.PREFIX, key, value)
                content += "{} -{} {}\n".format(self.PREFIX, key, value)
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
        self.set(**{"J": self._job_name})
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
        fout = p.stdout
        lines = fout.readlines()

        # - run over results
        """format
        JOBID   USER    STAT  QUEUE      FROM_HOST   EXEC_HOST   JOB_NAME   SUBMIT_TIME
        120727  jxu     PEND  normal     manage01                *ee.script Mar  1 21:55
        """
        finished = False
        for line in lines[1:]:  # skipe first info line
            data = line.strip().split()
            jobid, name, status = data[0], data[6], data[2]
            # if name.startswith(self.prefix) and status in self.running_status:
            #    indices = re.match(self.prefix+"*", name).span()
            #    if indices is not None:
            #        confid = int(name[indices[1]:])
            #    confids.append(int(confid))
            if name == self.job_name:
                finished = False
                break
        else:
            finished = True

        return finished


if __name__ == "__main__":
    ...
