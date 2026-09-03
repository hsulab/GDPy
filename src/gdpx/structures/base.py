"""Stable contracts for producing and transforming atomic structures."""

from typing import Any, Protocol, Sequence, runtime_checkable


@runtime_checkable
class StructureSource(Protocol):
    def load(self) -> Sequence[Any]:
        ...


@runtime_checkable
class StructureBuilder(Protocol):
    def run(self, substrates: Any = None, **context: Any) -> Sequence[Any]:
        ...


@runtime_checkable
class StructureModifier(Protocol):
    def run(self, substrates: Any = None, **context: Any) -> Sequence[Any]:
        ...
