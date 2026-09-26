"""PLUMED modifier and calculator provider."""

from ..adapters import manager_provider
from .plumed import PlumedManager

PLUMED_PROVIDER = manager_provider(
    "plumed", "gdpx.providers.plumed.plumed", "PlumedManager",
    {"ase.calculator": "ase"},
)

__all__ = ["PlumedManager", "PLUMED_PROVIDER"]
