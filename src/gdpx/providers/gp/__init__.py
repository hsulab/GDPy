"""Gaussian-process potential provider."""

from ..adapters import manager_provider
from .base import GP
from .calculator import GPCalculator
from .fgp import FGP
from .manager import GaussianProcessManager
from .serialization import load_model, save_model
from .sgp import SGP
from .trainer import GaussianProcessTrainer

GP_PROVIDER = manager_provider(
    "gp", "gdpx.providers.gp.manager", "GaussianProcessManager",
    {"ase.calculator": "ase"},
    trainer=("gdpx.providers.gp.trainer", "GaussianProcessTrainer"),
)

__all__ = ["GP", "FGP", "SGP", "GPCalculator", "GaussianProcessManager",
           "GaussianProcessTrainer", "save_model", "load_model", "GP_PROVIDER"]
