"""Capability identifiers and light-weight provider protocols."""

from __future__ import annotations

import enum
from typing import Any, Mapping, Protocol, runtime_checkable


class CapabilityKind(str, enum.Enum):
    POTENTIAL = "potential"
    MATERIALIZER = "materializer"
    EXECUTOR = "executor"
    TRAINER = "trainer"
    DATASET_CODEC = "dataset_codec"
    MODIFIER = "modifier"
    COLLECTIVE_VARIABLE = "collective_variable"
    SCHEDULER = "scheduler"
    EXPLORATION = "exploration"


@runtime_checkable
class Factory(Protocol):
    def create(self, parameters: Mapping[str, Any], **context: Any) -> Any:
        """Create one configured component."""


@runtime_checkable
class Materializer(Protocol):
    def materialize(self, potential: Any, target: type, **context: Any) -> Any:
        """Translate a neutral potential to a target representation."""

