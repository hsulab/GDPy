#!/usr/bin/env python3
# -*- coding: utf-8 -*

import copy
import pathlib
import subprocess
from typing import NoReturn, Union, List

from abc import ABC, abstractmethod


class AbstractScheduler(ABC):
    """The abstract scheduler that implements common functions.

    A scheduler deals with the lifecycle of a job in the queue.

    Attributes:
    """

    #: The name of the scheduler.
    name: str = "abstract"

    #: A string starts at each option line.
    PREFIX: str = None

    #: The suffix of a job script.
    SUFFIX: str = None

    #: The first line of a script.
    SHELL: str = None

    #: The command used to submit jobs.
    SUBMIT_COMMAND: str = None

    #: The command used to check job status.
    ENQUIRE_COMMAND: str = None

    #: Default parameters.
    default_parameters: dict = {}

    #: Current stored parameters.
    parameters: dict = {}

    #: _script: The path of the job script.
    _script: Union[str, pathlib.Path] = None

    #: The job name.
    _job_name: str = "scheduler"

    #: Environment settings for a job.
    environs: str = None

    #: Custom commands for a job.
    user_commands: str = None

    #: The tags that a job may have in the queue.
    running_status: List[str] = []

    def __init__(self, *args, **kwargs):
        """Init an abstract scheduler.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        """
        # - update params
        self.environs = kwargs.pop("environs", "")
        self.user_commands = kwargs.pop("user_commands", "")

        # - make default params
        self.parameters = self._get_default_parameters()
        # parameters_ = kwargs.pop("parameters", None)
        # if parameters_:
        #    self.parameters.update(parameters_)
        self.parameters.update(kwargs)

        # - update some special keywords
        #   job_name

        return

    @property
    def script(self) -> Union[str, pathlib.Path]:
        """Store the path of the job script."""

        return self._script

    @script.setter
    def script(self, script_):
        self._script = pathlib.Path(script_)
        return

    @property
    def job_name(self) -> str:

        return self._job_name

    @job_name.setter
    @abstractmethod
    def job_name(self, job_name_: str):
        self._job_name = job_name_
        # update job name in parameters
        return

    def _get_default_parameters(self):
        return copy.deepcopy(self.default_parameters)

    def set(self, **kwargs) -> None:
        """Set parameters.

        Args:
            **kwargs: Arbitrary keyword arguments.

        """
        # changed_parameters = {}
        for key, value in kwargs.items():
            oldvalue = self.parameters.get(key)
            # if key not in self.parameters or not equal(value, oldvalue):
            #    changed_parameters[key] = value
            #    self.parameters[key] = value
            self.parameters[key] = value

        return

    def write(self) -> None:
        """Write self to the path of the job script."""
        with open(self.script, "w") as fopen:
            fopen.write(str(self))

        return

    def submit(self) -> str:
        """Submit job using specific scheduler command and return job id."""
        command = "{0} {1}".format(self.SUBMIT_COMMAND, self.script.name)
        proc = subprocess.Popen(
            command,
            shell=True,
            cwd=self.script.parent,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            encoding="utf-8",
        )
        errorcode = proc.wait(timeout=10)  # 10 seconds
        if errorcode:
            raise RuntimeError(f"Error in submitting job script {str(self.script)}")

        output = "".join(proc.stdout.readlines())
        job_id = output.strip().split()[-1]

        return job_id

    @abstractmethod
    def is_finished(self) -> bool:
        """Check whether the job is finished.

        The job is the one performed in the current job script path.

        """

        return

    def as_dict(self) -> dict:
        """"""
        sch_params = {}
        sch_params = {k: v for k, v in self.parameters.items() if v is not None}
        sch_params["environs"] = self.environs
        sch_params["backend"] = self.name

        sch_params = copy.deepcopy(sch_params)

        return sch_params


if __name__ == "__main__":
    ...
