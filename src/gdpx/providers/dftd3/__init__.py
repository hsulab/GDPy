"""DFT-D3 standalone potential and additive ASE modifier."""

from ..adapters import BackendFactory, ManagerModifierFactory, add_capabilities, manager_provider
from ..capabilities import CapabilityKind
from .manager import Dftd3Manager

DFTD3_PROVIDER = manager_provider(
    "dftd3", "gdpx.providers.dftd3.manager", "Dftd3Manager",
    {"ase.calculator": "ase"},
)
DFTD3_PROVIDER = add_capabilities(
    DFTD3_PROVIDER, CapabilityKind.MODIFIER,
    {"default": BackendFactory("ase", {
        "ase": ManagerModifierFactory("gdpx.providers.dftd3.manager", "Dftd3Manager"),
    })},
)

__all__ = ["Dftd3Manager", "DFTD3_PROVIDER"]
