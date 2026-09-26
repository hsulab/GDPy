"""Backend-neutral immutable specifications shared by providers."""

from __future__ import annotations

import pathlib
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Optional, Tuple


def freeze(value: Any) -> Any:
    """Recursively freeze ordinary configuration containers."""
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(freeze(item) for item in value)
    if isinstance(value, set):
        return frozenset(freeze(item) for item in value)
    return value


def thaw(value: Any) -> Any:
    """Return JSON/YAML-friendly mutable copies of frozen containers."""
    if isinstance(value, Mapping):
        return {key: thaw(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [thaw(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return [thaw(item) for item in value]
    if isinstance(value, pathlib.Path):
        return str(value)
    return value


@dataclass(frozen=True)
class Artifact:
    path: pathlib.Path
    role: str
    digest: Optional[str] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", pathlib.Path(self.path))
        if not self.role:
            raise ValueError("Artifact role cannot be empty.")


@dataclass(frozen=True)
class PotentialSpec:
    provider: str
    parameters: Mapping[str, Any] = field(default_factory=dict)
    version: Optional[str] = None
    method: str = "default"
    backend: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.provider or not self.method:
            raise ValueError("Potential provider and method cannot be empty.")
        object.__setattr__(self, "parameters", freeze(self.parameters))


@dataclass(frozen=True)
class ModifierSpec:
    provider: str
    method: str
    parameters: Mapping[str, Any] = field(default_factory=dict)
    backend: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.provider or not self.method:
            raise ValueError("Modifier provider and method cannot be empty.")
        object.__setattr__(self, "parameters", freeze(self.parameters))


@dataclass(frozen=True)
class Materialization:
    target: str
    payload: Any
    artifacts: Tuple[Artifact, ...] = ()

    def __post_init__(self) -> None:
        if not self.target:
            raise ValueError("Materialization target cannot be empty.")
        object.__setattr__(self, "artifacts", tuple(self.artifacts))


@dataclass(frozen=True)
class TrainingSpec:
    provider: str
    parameters: Mapping[str, Any] = field(default_factory=dict)
    artifacts: Tuple[Artifact, ...] = ()
    method: str = "default"

    def __post_init__(self) -> None:
        if not self.provider or not self.method:
            raise ValueError("Training provider and method cannot be empty.")
        object.__setattr__(self, "parameters", freeze(self.parameters))
        object.__setattr__(self, "artifacts", tuple(self.artifacts))
