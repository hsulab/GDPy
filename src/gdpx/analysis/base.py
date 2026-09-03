"""Stable analysis contracts independent of execution implementations."""

import abc
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class FeatureSet:
    values: Any
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Selection:
    structures: Sequence[Any]
    indices: Sequence[int]
    metadata: Mapping[str, Any] = field(default_factory=dict)


class Descriptor(abc.ABC):
    @abc.abstractmethod
    def describe(self, structures: Sequence[Any]) -> FeatureSet:
        ...


class Comparator(abc.ABC):
    @abc.abstractmethod
    def compare(self, first: Any, second: Any) -> Any:
        ...


class Selector(abc.ABC):
    @abc.abstractmethod
    def select(self, structures: Sequence[Any], features: Any = None) -> Selection:
        ...


class Validator(abc.ABC):
    @abc.abstractmethod
    def validate(self, reference: Any, prediction: Any):
        ...

