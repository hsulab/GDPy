#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from enum import Enum, auto


class Pairing(Enum):
    """How to pair N drivers with M structures when building the task grid."""

    BROADCAST = auto()
    """1 driver → all M structures (original DriverBasedWorker mode)."""

    REPEAT = auto()
    """N drivers ← 1 structure (broadcast a single struct to all drivers)."""

    BIJECTION = auto()
    """N drivers × N structures, 1:1 pairing.  Requires N == M."""

    PRODUCT = auto()
    """Full cartesian product: N drivers × M structures."""

    PARTITION = auto()
    """Split M structures across N drivers (HPC scatter pattern)."""

    AUTO = auto()
    """Infer from N vs M counts."""

    @classmethod
    def _missing_(cls, value):
        """Allow lookup by case-insensitive name (e.g. ``Pairing('auto')``)."""
        if isinstance(value, str):
            for member in cls:
                if member.name.lower() == value.lower():
                    return member
        return super()._missing_(value)

    def __str__(self) -> str:
        return self.name


if __name__ == "__main__":
    pass
