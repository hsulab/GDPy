"""nequip software provider."""

from ..adapters import manager_provider
from .manager import NequipManager

NEQUIP_PROVIDER = manager_provider(
    "nequip",
    "gdpx.providers.nequip.manager",
    "NequipManager",
    {"ase.calculator":"ase","lammps.potential":"lammps"},
    trainer=("gdpx.providers.nequip.trainer", "NequipTrainer"),
)

__all__ = ["NequipManager", "NEQUIP_PROVIDER"]

