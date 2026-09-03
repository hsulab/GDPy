"""reax software provider."""

from ..adapters import manager_provider
from .manager import ReaxManager

REAX_PROVIDER = manager_provider(
    "reax",
    "gdpx.providers.reax.manager",
    "ReaxManager",
    {"lammps.potential":"lammps"},
)

__all__ = ["ReaxManager", "REAX_PROVIDER"]

