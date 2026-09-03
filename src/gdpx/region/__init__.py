"""Deprecated compatibility namespace for :mod:`gdpx.structures.regions`."""

import warnings

warnings.warn(
    "gdpx.region is deprecated; import from gdpx.structures.regions.",
    DeprecationWarning,
    stacklevel=2,
)

from gdpx.structures.regions import (  # noqa: E402,F401
    AutoRegion, CubeRegion, CylinderRegion, LatticeRegion, REGISTER, SphereRegion,
)

__all__ = ["REGISTER", "AutoRegion", "CubeRegion", "CylinderRegion", "LatticeRegion", "SphereRegion"]
