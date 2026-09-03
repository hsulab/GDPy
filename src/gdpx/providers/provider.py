"""The stateless provider descriptor."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping

from .capabilities import CapabilityKind


@dataclass(frozen=True)
class Provider:
    """A named integration exposing factories through capability maps."""

    name: str
    version: str = "unknown"
    capabilities: Mapping[CapabilityKind, Mapping[str, object]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Provider name cannot be empty.")
        normalized = {}
        for raw_kind, implementations in self.capabilities.items():
            kind = raw_kind if isinstance(raw_kind, CapabilityKind) else CapabilityKind(raw_kind)
            normalized[kind] = MappingProxyType(dict(implementations))
        object.__setattr__(self, "capabilities", MappingProxyType(normalized))

    def implementations(self, kind: CapabilityKind) -> Mapping[str, object]:
        return self.capabilities.get(kind, MappingProxyType({}))

