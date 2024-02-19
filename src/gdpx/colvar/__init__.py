#!/usr/bin/env python3
# -*- coding: utf-8 -*-


try:
    import jax
except Exception as e:
    ...

from ..core.register import registers

from .distance import DistanceColvar
registers.colvar.register("DistanceColvar")(DistanceColvar)

from .fingerprint import FingerprintColvar
registers.colvar.register("FingerprintColvar")(FingerprintColvar)

from .position import position
registers.colvar.register("position")(position)


if __name__ == "__main__":
    ...