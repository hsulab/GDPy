"""dftd4 software provider."""

from ..adapters import manager_provider
from .manager import Dftd4Manager

DFTD4_PROVIDER = manager_provider(
    "dftd4",
    "gdpx.providers.dftd4.manager",
    "Dftd4Manager",
    {"ase.calculator":"ase"},
)

__all__ = ["Dftd4Manager", "DFTD4_PROVIDER"]

