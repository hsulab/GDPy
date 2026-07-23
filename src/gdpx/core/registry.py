"""Small, dependency-free implementation registry."""

from __future__ import annotations

import warnings
from collections.abc import Callable, Iterator
from typing import Any


class Registry:
    """Map configuration names to callable implementations."""

    def __init__(self, name: str) -> None:
        self._dict: dict[str, Callable[..., Any]] = {}
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def __setitem__(self, key: str | None, value: Callable[..., Any]) -> None:
        if not callable(value):
            raise TypeError(f"Value of registry {self._name!r} must be callable: {value!r}")
        resolved_key = value.__name__ if key is None else key
        if resolved_key in self._dict:
            warnings.warn(
                f"Key {resolved_key} already in registry {self._name}.",
                UserWarning,
                stacklevel=2,
            )
        self._dict[resolved_key] = value

    def register(self, target):
        """Register a callable directly or return a named decorator."""

        def add(key, value):
            self[key] = value
            return value

        if callable(target):
            return add(None, target)
        return lambda value: add(target, value)

    def __getitem__(self, key: str) -> Callable[..., Any]:
        if key not in self._dict:
            raise KeyError(f"No {key!r} in {self._name} registry; available: {tuple(self._dict)}")
        return self._dict[key]

    def __contains__(self, key: object) -> bool:
        return key in self._dict

    def keys(self) -> Iterator[str]:
        return iter(self._dict.keys())

    def __repr__(self) -> str:
        content = f"{self._name.upper()}:\n"
        keys = sorted(self._dict)
        for offset in range(0, len(keys), 5):
            row = keys[offset : offset + 5]
            content += ("  " + "{:<24s}" * len(row) + "\n").format(*row)
        return content


# Historical names retained for downstream plugins.
Register = Registry
BaseRegister = Registry
