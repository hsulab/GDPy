"""eam software provider."""

from ..adapters import manager_provider
from .manager import EamManager

EAM_PROVIDER = manager_provider(
    "eam",
    "gdpx.providers.eam.manager",
    "EamManager",
    {"ase.calculator":"ase","lammps.potential":"lammps"},
)

__all__ = ["EamManager", "EAM_PROVIDER"]

