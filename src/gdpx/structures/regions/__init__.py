"""Spatial region implementations and registry."""

from .registry import REGION_REGISTRY as REGISTER
from .region import AutoRegion, CubeRegion, CylinderRegion, LatticeRegion, SphereRegion

REGISTER.register(AutoRegion)
REGISTER.register(CubeRegion)
REGISTER.register(SphereRegion)
REGISTER.register(CylinderRegion)
REGISTER.register(LatticeRegion)

__all__ = ["AutoRegion", "CubeRegion", "CylinderRegion", "LatticeRegion", "REGISTER", "SphereRegion"]
