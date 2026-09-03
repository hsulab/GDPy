"""Contracts for modifiers and collective variables."""

import abc
from typing import Any


class CollectiveVariable(abc.ABC):
    @abc.abstractmethod
    def evaluate(self, structure: Any) -> Any:
        ...


class Modifier(abc.ABC):
    @abc.abstractmethod
    def evaluate(self, structure: Any) -> Any:
        """Return the modifier's energy/force contribution or state."""

