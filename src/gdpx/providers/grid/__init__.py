"""grid software provider."""

from ..adapters import ExecutorFactory, add_capabilities, manager_provider
from ..capabilities import CapabilityKind
from .manager import GridManager

GRID_PROVIDER = manager_provider(
    "grid",
    "gdpx.providers.grid.manager",
    "GridManager",
    {"ase.calculator":"grid"},
)
GRID_PROVIDER = add_capabilities(
    GRID_PROVIDER,
    CapabilityKind.EXECUTOR,
    {"neb": ExecutorFactory(
        "gdpx.providers.grid.path", "ZeroStringReactor", "neb", "ase.calculator"
    )},
)

__all__ = ["GridManager", "GRID_PROVIDER"]
