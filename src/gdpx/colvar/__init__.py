#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from gdpx.core.register import BaseRegister

REGISTER = BaseRegister("colvar")

from .distance import DistanceColvar

REGISTER.register("DistanceColvar")(DistanceColvar)

from .rmsd import RmsdColvar

REGISTER.register("RmsdColvar")(RmsdColvar)

from .fingerprint import FingerprintColvar

REGISTER.register("FingerprintColvar")(FingerprintColvar)

from .position import position

REGISTER.register("position")(position)


if __name__ == "__main__":
    ...

