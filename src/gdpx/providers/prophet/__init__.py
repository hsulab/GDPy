"""Prophet software provider; optional dependencies load on materialization."""

from ..adapters import manager_provider
from .manager import ProphetManager

PROPHET_PROVIDER = manager_provider(
    "prophet",
    "gdpx.providers.prophet.manager",
    "ProphetManager",
    {"ase.calculator": "ase"},
)

__all__ = ["ProphetManager", "PROPHET_PROVIDER"]
