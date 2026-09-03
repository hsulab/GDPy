"""Typed scientific results returned by executors and training services."""

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence


@dataclass
class ExecutionResult:
    converged: bool
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationResult(ExecutionResult):
    structure: Any = None


@dataclass
class TrajectoryResult(ExecutionResult):
    trajectory: Sequence[Any] = ()


@dataclass
class OptimizationResult(TrajectoryResult):
    structure: Any = None


@dataclass
class TransitionStateResult(OptimizationResult):
    mode: Any = None
    curvature: Optional[float] = None


@dataclass
class PathwayResult(ExecutionResult):
    images: Sequence[Any] = ()
    trajectories: Sequence[Sequence[Any]] = ()
    transition_states: Sequence[Any] = ()


@dataclass
class TrainingResult:
    model: Any
    converged: bool
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class ValidationReport:
    passed: bool
    metrics: Mapping[str, Any] = field(default_factory=dict)
    artifacts: Sequence[Any] = ()

