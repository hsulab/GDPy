"""ReaxFF provider: xreac for ASE and reax/c for LAMMPS."""

from ..adapters import manager_provider
from .manager import ReaxManager

REAX_PROVIDER = manager_provider(
    "reax",
    "gdpx.providers.reax.manager",
    "ReaxManager",
    {"ase.calculator": ("xreac", "reax/c"), "lammps.potential": "reax/c"},
)

__all__ = ["ReaxManager", "REAX_PROVIDER"]
