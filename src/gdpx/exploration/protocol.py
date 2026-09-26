"""Boundary between adaptive scientific search and task execution."""

import abc
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class Proposal:
    inputs: Any
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExplorationResult:
    structures: Sequence[Any]
    converged: bool
    metadata: Mapping[str, Any] = field(default_factory=dict)


class ExplorationStrategy(abc.ABC):
    """Own scientific proposals and acceptance, never job execution."""

    @abc.abstractmethod
    def propose(self) -> Proposal:
        ...

    @abc.abstractmethod
    def observe(self, proposal: Proposal, result: Any) -> None:
        ...

    @abc.abstractmethod
    def converged(self) -> bool:
        ...


class Exploration:
    """Drive an adaptive search through an injected execution service."""

    def __init__(self, strategy: ExplorationStrategy, execution_service: Any, runtime: Any) -> None:
        self.strategy = strategy
        self.execution_service = execution_service
        self.runtime = runtime

    def step(self):
        proposal = self.strategy.propose()
        handle = self.execution_service.submit(self.runtime, proposal.inputs)
        result = self.execution_service.retrieve(handle)
        self.strategy.observe(proposal, result)
        return result

