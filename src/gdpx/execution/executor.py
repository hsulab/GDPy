"""Scientific executor contracts."""

import abc
from typing import Any
from dataclasses import dataclass
from enum import Enum

from gdpx.core.component import BaseComponent


class RunState(Enum):
    """An executor may yield while a child calculation is pending."""

    PENDING = "pending"
    FINISHED = "finished"


@dataclass
class ExecutionSettings:
    machine_prefix: str = ""
    task: str = ""


class SingleStructureExecutor(BaseComponent):
    """Calculator-independent contract consumed by computation workers."""

    setting: ExecutionSettings

    @abc.abstractmethod
    def run(self, atoms, read_ckpt=True, **kwargs): ...

    @abc.abstractmethod
    def reset(self): ...

    @abc.abstractmethod
    def read_convergence(self): ...

    @abc.abstractmethod
    def read_trajectory(self, **kwargs): ...


class Executor(abc.ABC):
    """Perform one explicitly requested scientific calculation."""

    method = "execute"

    @abc.abstractmethod
    def run(self, inputs: Any, **kwargs: Any) -> Any:
        ...


class EvaluationExecutor(Executor):
    pass


class OptimizationExecutor(Executor):
    pass


class DynamicsExecutor(Executor):
    pass


class TransitionStateExecutor(Executor):
    pass


class LocalTransitionStateExecutor(TransitionStateExecutor):
    """Find a saddle point from one initial structure and optional mode."""


class PathTransitionStateExecutor(TransitionStateExecutor):
    """Find a pathway from endpoints or an initial image sequence."""

