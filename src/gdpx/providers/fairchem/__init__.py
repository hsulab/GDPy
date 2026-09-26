"""fairchem software provider."""

from ..adapters import manager_provider
from .manager import FairChemManager

FAIRCHEM_PROVIDER = manager_provider(
    "fairchem",
    "gdpx.providers.fairchem.manager",
    "FairChemManager",
    {"ase.calculator":"ase"},
)

__all__ = ["FairChemManager", "FAIRCHEM_PROVIDER"]

