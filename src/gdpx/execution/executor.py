"""Scientific executor contracts."""

import abc
from typing import Any


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

