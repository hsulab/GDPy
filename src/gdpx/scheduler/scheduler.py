#!/usr/bin/env python3
# -*- coding: utf-8 -*

import copy
import pathlib
import subprocess
from typing import Optional, Union, List, Callable, Iterable

from abc import ABC, abstractmethod

from .. import config


class AbstractScheduler(ABC):
    """The abstract scheduler that implements common functions.

    A scheduler deals with the lifecycle of a job in the queue.

    Attributes:
    """

    #: The name of the scheduler.
    name: str = "abstract"

    #: Standard print function.
    _print: Callable = config._print

    #: Standard debug function.
    _debug: Callable = config._debug

    #: Host name.
    hostname: str = "local"

    #: A string starts at each option line.
    PREFIX: str = ""

    #: The suffix of a job script.
    SUFFIX: str = ""

    #: The first line of a script.
    SHELL: str = ""

    #: The command used to submit jobs.
    SUBMIT_COMMAND: str = ""

    #: The command used to check job status.
    ENQUIRE_COMMAND: str = ""

    #: Default parameters.
    default_parameters: dict = {}

    #: Current stored parameters.
    parameters: dict = {}

    #: The path of the job script.
    _script: Union[str, pathlib.Path] = "./run.script"

    #: The job name.
    _job_name: str = "scheduler"

    #: Environment settings for a job.
    environs: Union[str, List[str]] = ""

    #: Machine-related prefix added before executable (e.g. mpirun).
    machine_prefix : str = ""

    #: Custom commands for a job.
    user_commands: str = ""

    #: The tags that a job may have in the queue.
    running_status: List[str] = []

    def __init__(self, *args, **kwargs):
        """Init an abstract scheduler.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        """
        # update params
        self.environs = kwargs.pop("environs", "")
        self.machine_prefix = kwargs.pop("machine_prefix", "")
        self.user_commands = kwargs.pop("user_commands", "")

        self.hostname = kwargs.pop("hostname", "local")
        self.remote_wdir = kwargs.pop("remote_wdir", "./")

        # make default params
        self.parameters = self._get_default_parameters()
        # parameters_ = kwargs.pop("parameters", None)
        # if parameters_:
        #    self.parameters.update(parameters_)
        self.parameters.update(kwargs)

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

    def _convert_environs_to_content(self) -> str:
        """"""
        content = "\n\n"
        if self.environs:
            if isinstance(self.environs, str):
                content += self.environs
            elif isinstance(self.environs, Iterable):
                for env in self.environs:
                    content += env.strip() + "\n"
            else:
                raise RuntimeError(f"Fail to convert environs `{self.environs}`.")
        else:
            ...
        content += "\n\n"

        return content

    def write(self) -> None:
        """Write self to the path of the job script."""
        with open(self.script, "w") as fopen:
            fopen.write(str(self))

        return

    def submit(self) -> str:
        """Submit job using specific scheduler command and return job id."""
        assert isinstance(self.script, pathlib.Path)
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

        output = "".join(proc.stdout.readlines())  # type: ignore
        job_id = output.strip().split()[-1]

        return job_id

    @abstractmethod
    def is_finished(self) -> bool:
        """Check whether the job is finished.

        The job is the one performed in the current job script path.

        """

        ...

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
