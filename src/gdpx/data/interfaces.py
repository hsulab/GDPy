"""Separated dataset loading, encoding, and splitting contracts."""

import abc
from typing import Any, Mapping, Sequence


class DatasetSource(abc.ABC):
    @abc.abstractmethod
    def load_frames(self) -> Any:
        ...


class DatasetCodec(abc.ABC):
    @abc.abstractmethod
    def encode(self, dataset: Any, destination: Any) -> Any:
        ...

    @abc.abstractmethod
    def decode(self, source: Any) -> Any:
        ...


class DatasetSplitter(abc.ABC):
    @abc.abstractmethod
    def split(self, dataset: Any) -> Mapping[str, Sequence[Any]]:
        ...
