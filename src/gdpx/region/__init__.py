#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from gdpx.core.register import BaseRegister

REGISTER = BaseRegister("region")

from .region import AutoRegion, CubeRegion, CylinderRegion, LatticeRegion, SphereRegion

REGISTER.register(AutoRegion)
REGISTER.register(CubeRegion)
REGISTER.register(SphereRegion)
REGISTER.register(CylinderRegion)
REGISTER.register(LatticeRegion)

__all__ = ["REGISTER", "AutoRegion", "CubeRegion", "CylinderRegion", "LatticeRegion", "SphereRegion"]


if __name__ == "__main__":
    ...
