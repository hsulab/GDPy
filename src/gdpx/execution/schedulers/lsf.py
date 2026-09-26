import re
import subprocess

from .scheduler import BaseScheduler


class LsfScheduler(BaseScheduler):
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

        content += self._convert_environs_to_content()

        if self.user_commands:
            content += "\n\n"
            content += self.user_commands

        return content

    @BaseScheduler.job_name.setter
    def job_name(self, job_name_: str):
        self._job_name = job_name_
        self.set(**{"J": self._job_name})
        return

    def parse_submit_output(self, output: str) -> str:
        """Extract the identifier from ``Job <123> is submitted ...``."""
        match = re.search(r"Job\s+<(\d+)>", output)
        if match is None:
            raise RuntimeError(f"Cannot parse LSF submission output: {output.strip()!r}")
        return match.group(1)

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
        return self.is_finished_from_output("".join(lines))

    def is_finished_from_output(self, output: str) -> bool:
        """Return whether this job name is absent from ``bjobs -w`` output."""
        for line in output.splitlines()[1:]:
            fields = line.split()
            if len(fields) >= 7 and fields[6] == self.job_name:
                return False
        return True
