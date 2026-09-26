"""Stable materialization targets shared by providers and executors."""

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from .specs import Artifact


@dataclass(frozen=True)
class AseCalculatorMaterialization:
    calculator: Any
    artifacts: Sequence[Artifact] = ()


@dataclass(frozen=True)
class LammpsPotentialMaterialization:
    commands: Sequence[str]
    artifacts: Sequence[Artifact] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    calculator: Any = None


@dataclass(frozen=True)
class NativeInputMaterialization:
    provider: str
    parameters: Mapping[str, Any]
    artifacts: Sequence[Artifact] = ()

