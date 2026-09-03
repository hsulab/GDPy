"""Built-in bias calculator provider."""

from ..adapters import manager_provider
from .manager import BiasManager

BIAS_PROVIDER = manager_provider(
    "bias", "gdpx.providers.bias.manager", "BiasManager",
    {"ase.calculator": "ase"},
)

__all__ = ["BiasManager", "BIAS_PROVIDER"]
