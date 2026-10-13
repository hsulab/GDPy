"""Compatibility imports for the former monolithic region module."""

from .base import BaseRegion
from .auto import AutoRegion
from .cube import CubeRegion
from .sphere import SphereRegion
from .cylinder import CylinderRegion
from .lattice import LatticeRegion
from .surface_lattice import SurfaceLatticeRegion
from .surface import SurfaceRegion
from .intersect import IntersectRegion
from .hemisphere import HemisphereRegion

__all__ = [
    "BaseRegion",
    "AutoRegion",
    "CubeRegion",
    "SphereRegion",
    "CylinderRegion",
    "LatticeRegion",
    "SurfaceLatticeRegion",
    "SurfaceRegion",
    "IntersectRegion",
    "HemisphereRegion",
]
