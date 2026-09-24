"""xtb software provider."""

from ..adapters import manager_provider
from .manager import XtbManager

XTB_PROVIDER = manager_provider(
    "xtb",
    "gdpx.providers.xtb.manager",
    "XtbManager",
    {"ase.calculator": ("xtb", "tblite")},
)

__all__ = ["XtbManager", "XTB_PROVIDER"]

