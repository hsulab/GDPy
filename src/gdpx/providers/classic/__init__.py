"""classic software provider."""

from ..adapters import manager_provider
from .manager import ClassicManager

CLASSIC_PROVIDER = manager_provider(
    "classic",
    "gdpx.providers.classic.manager",
    "ClassicManager",
    {"lammps.potential":"lammps"},
)

__all__ = ["ClassicManager", "CLASSIC_PROVIDER"]

