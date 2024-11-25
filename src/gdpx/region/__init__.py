#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from ..core.register import registers
from .region import (AutoRegion, CubeRegion, CylinderRegion, IntersectRegion,
                     LatticeRegion, SphereRegion, SurfaceLatticeRegion,
                     SurfaceRegion)

registers.region.register(AutoRegion)
registers.region.register(CubeRegion)
registers.region.register(SphereRegion)
registers.region.register(CylinderRegion)
registers.region.register(LatticeRegion)

# BUG: periodic boundary along z-axis
registers.region.register(SurfaceLatticeRegion)

# BUG: periodic boundary along z-axis
registers.region.register(SurfaceRegion)

registers.region.register(IntersectRegion)


if __name__ == "__main__":
    ...
