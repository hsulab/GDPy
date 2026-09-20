import abc
import copy
import pathlib
import subprocess
from typing import Callable, Iterable, Optional, Union

from gdpx import config


def submit_job_script(
    script_fpath: pathlib.Path,
    submit_command: str,
    submit_timeout: float,
    is_dry_run: bool = False,
    parse_output: Optional[Callable[[str], str]] = None,
) -> str:
    """Submit job script."""
    command = f"{submit_command} {script_fpath.name}"
    if not is_dry_run:
        proc = subprocess.Popen(
            command,
            shell=True,
            cwd=script_fpath.parent,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            encoding="utf-8",
        )
        errorcode = proc.wait(timeout=submit_timeout)
        if errorcode:
            raise RuntimeError(f"Error in submitting job script {str(script_fpath)}")

        output = "".join(proc.stdout.readlines())  # type: ignore
        if not output.strip():
            raise RuntimeError(f"Scheduler returned no job id for {str(script_fpath)}")
        job_id = (parse_output or (lambda value: value.strip().split()[-1]))(output)
    else:
        job_id = f"Attempt to submit the job script `{script_fpath.name}` with command `{command}`."

    return job_id


class BaseScheduler(abc.ABC):
    """The abstract scheduler that implements common functions.

    A scheduler deals with the lifecycle of a job in the queue.

    Attributes:
    """

    #: The name of the scheduler.
    name: str = "abstract"

    #: Whether jobs execute without a queue manager.
    is_direct: bool = False

    #: Transport used to reach the execution host.
    transport_name: str = "local"

    #: Standard print function.
    _print: Callable = config._print

    #: Standard debug function.
    _debug: Callable = config._debug

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

    #: The job name.
    _job_name: str = "scheduler"

    #: Environment settings for a job.
    environs: Union[str, list[str]] = ""

    #: Machine-related prefix added before executable (e.g. mpirun).
    machine_prefix: str = ""

    #: Custom commands for a job.
    user_commands: str = ""

    #: The tags that a job may have in the queue.
    running_status: list[str] = []

    def __init__(self, submit_timeout: float = 10.0, is_dry_run: bool = False, *args, **kwargs):
        """Init an abstract scheduler.

        Args:
            submit_timeout: Timeout for running the submit command.
            is_dry_run: Whether submit the job (for test).
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        """
        # basic params
        self.submit_timeout = submit_timeout
        self.is_dry_run = is_dry_run

        # update params
        self.environs = kwargs.pop("environs", "")
        self.machine_prefix = kwargs.pop("machine_prefix", "")
        self.user_commands = kwargs.pop("user_commands", "")

        # make default params
        self.parameters = self._get_default_parameters()
        # parameters_ = kwargs.pop("parameters", None)
        # if parameters_:
        #    self.parameters.update(parameters_)
        self.parameters.update(kwargs)

        # Some default settings

        #: The path of the job script.
        self._script: pathlib.Path = pathlib.Path("./run.script")

        return

    @property
    def script(self) -> pathlib.Path:
        """Store the path of the job script."""

        return self._script

    @script.setter
    def script(self, script: Union[str, pathlib.Path]) -> None:
        self._script = pathlib.Path(script)

        return

    @property
    def job_name(self) -> str:

        return self._job_name

    @job_name.setter
    @abc.abstractmethod
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

    def build_submit_command(self, script_name: str) -> str:
        """Build the shell command that submits *script_name*."""
        return f"{self.SUBMIT_COMMAND} {script_name}"

    def parse_submit_output(self, output: str) -> str:
        """Extract a scheduler job identifier from submission output."""
        if not output.strip():
            raise RuntimeError(f"{self.name} returned empty submission output.")
        return output.strip().split()[-1]

    def is_finished_from_output(self, output: str) -> bool:
        """Interpret queue enquiry output produced on any host.

        Queue schedulers that can be wrapped by an SSH transport implement
        this hook. It is deliberately non-abstract so existing third-party
        schedulers remain usable locally.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support transport-independent status parsing."
        )

    def sync(self, wdir_names: Iterable[str] = ()) -> None:
        """Synchronize completed job data; local transports have nothing to do."""
        return

    def submit(self, func_to_execute: Optional[Callable] = None) -> str:
        """Submit job using specific scheduler command and return job id."""
        if func_to_execute is None:  # compatible mode
            assert isinstance(self.script, pathlib.Path)
            if not self.script.exists():
                self.write()
            else:
                ...
            job_id = submit_job_script(
                self.script,
                submit_command=self.SUBMIT_COMMAND,
                submit_timeout=self.submit_timeout,
                is_dry_run=self.is_dry_run,
                parse_output=self.parse_submit_output,
            )
        else:
            job_id = "direct"
            if self.is_direct:
                func_to_execute()
            else:
                assert isinstance(self.script, pathlib.Path)
                if not self.script.exists():
                    self.write()
                else:
                    ...
                job_id = submit_job_script(
                    self.script,
                    submit_command=self.SUBMIT_COMMAND,
                    submit_timeout=self.submit_timeout,
                    is_dry_run=self.is_dry_run,
                    parse_output=self.parse_submit_output,
                )

        return job_id

    @abc.abstractmethod
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
        sch_params["machine_prefix"] = self.machine_prefix
        sch_params["submit_timeout"] = self.submit_timeout
        sch_params["is_dry_run"] = self.is_dry_run
        return {
            "provider": self.name,
            "parameters": copy.deepcopy(sch_params),
        }
