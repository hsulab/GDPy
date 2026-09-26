"""tace software provider."""

from ..adapters import manager_provider
from .manager import TaceManager

TACE_PROVIDER = manager_provider(
    "tace",
    "gdpx.providers.tace.manager",
    "TaceManager",
    {"ase.calculator":"ase"},
)

__all__ = ["TaceManager", "TACE_PROVIDER"]

