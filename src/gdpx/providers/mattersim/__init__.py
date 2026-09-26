"""mattersim software provider."""

from ..adapters import manager_provider
from .manager import MatterSimManager

MATTERSIM_PROVIDER = manager_provider(
    "mattersim",
    "gdpx.providers.mattersim.manager",
    "MatterSimManager",
    {"ase.calculator": ("ase", "graph_pes")},
)

__all__ = ["MatterSimManager", "MATTERSIM_PROVIDER"]

