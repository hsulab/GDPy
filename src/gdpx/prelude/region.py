#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from gdpx.core.register import registers
from gdpx.region.region import (
    AutoRegion,
    CubeRegion,
    CylinderRegion,
    LatticeRegion,
    SphereRegion,
)

registers.region.register(AutoRegion)
registers.region.register(CubeRegion)
registers.region.register(SphereRegion)
registers.region.register(CylinderRegion)
registers.region.register(LatticeRegion)


if __name__ == "__main__":
    ...
