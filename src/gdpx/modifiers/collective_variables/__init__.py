#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from gdpx.core.register import BaseRegister

REGISTER = BaseRegister("colvar")

REGISTER.register_lazy("DistanceColvar", "gdpx.modifiers.collective_variables.distance", "DistanceColvar")
REGISTER.register_lazy("RmsdColvar", "gdpx.modifiers.collective_variables.rmsd", "RmsdColvar")
REGISTER.register_lazy("FingerprintColvar", "gdpx.modifiers.collective_variables.fingerprint", "FingerprintColvar")
REGISTER.register_lazy("position", "gdpx.modifiers.collective_variables.position", "position")


if __name__ == "__main__":
    ...
