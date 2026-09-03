"""dftd3 software provider."""

from ..adapters import manager_provider
from .manager import Dftd3Manager

DFTD3_PROVIDER = manager_provider(
    "dftd3",
    "gdpx.providers.dftd3.manager",
    "Dftd3Manager",
    {"ase.calculator":"ase"},
)

__all__ = ["Dftd3Manager", "DFTD3_PROVIDER"]

