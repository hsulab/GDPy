#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from gdpx.core.register import BaseRegister

REGISTER = BaseRegister("colvar")

REGISTER.register_lazy("DistanceColvar", "gdpx.colvar.distance", "DistanceColvar")
REGISTER.register_lazy("RmsdColvar", "gdpx.colvar.rmsd", "RmsdColvar")
REGISTER.register_lazy("FingerprintColvar", "gdpx.colvar.fingerprint", "FingerprintColvar")
REGISTER.register_lazy("position", "gdpx.colvar.position", "position")


if __name__ == "__main__":
    ...
