"""Spatial region implementations and registry."""

from .registry import REGION_REGISTRY as REGISTER
from .auto import AutoRegion
from .cube import CubeRegion
from .cylinder import CylinderRegion
from .hemisphere import HemisphereRegion
from .lattice import LatticeRegion
from .sphere import SphereRegion
from .surface import SurfaceRegion

REGISTER.register(AutoRegion)
REGISTER.register(CubeRegion)
REGISTER.register(SphereRegion)
REGISTER.register(CylinderRegion)
REGISTER.register(LatticeRegion)
REGISTER.register(HemisphereRegion)
REGISTER.register(SurfaceRegion)

__all__ = ["AutoRegion", "CubeRegion", "CylinderRegion", "HemisphereRegion", "LatticeRegion", "REGISTER", "SphereRegion", "SurfaceRegion"]
