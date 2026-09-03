"""REANN and embedded-atom neural-network providers."""

from ..adapters import TrainerFactory, add_capabilities, manager_provider
from ..capabilities import CapabilityKind
from .beann import BeannManager
from .reann import ReannManager

BEANN_PROVIDER = manager_provider(
    "beann", "gdpx.providers.reann.beann", "BeannManager",
    {"ase.calculator": "ase", "lammps.potential": "lammps"},
    trainer=("gdpx.providers.reann.training.beann", "BeannTrainer"),
)
REANN_PROVIDER = manager_provider(
    "reann", "gdpx.providers.reann.reann", "ReannManager",
    {"ase.calculator": "ase"},
    trainer=("gdpx.providers.reann.training.reann", "ReannTrainer"),
)
BEANN_PROVIDER = add_capabilities(
    BEANN_PROVIDER,
    CapabilityKind.DATASET_CODEC,
    {"default": TrainerFactory("gdpx.providers.reann.data", "ReannDataloader")},
)
REANN_PROVIDER = add_capabilities(
    REANN_PROVIDER,
    CapabilityKind.DATASET_CODEC,
    {"default": TrainerFactory("gdpx.providers.reann.data", "ReannDataloader")},
)

__all__ = ["BeannManager", "ReannManager", "BEANN_PROVIDER", "REANN_PROVIDER"]
