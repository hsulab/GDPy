"""Portable continuation values produced by exploration implementations."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ExplorationContinuation:
    """Provider-owned state passed between exploration iterations."""

    provider: str
    metadata: Mapping[str, Any] = field(default_factory=dict)
    artifacts: tuple[str, ...] = ()
