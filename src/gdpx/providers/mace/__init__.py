"""mace software provider."""

from ..adapters import TrainerFactory, add_capabilities, manager_provider
from ..capabilities import CapabilityKind
from .manager import MaceManager

MACE_PROVIDER = manager_provider(
    "mace",
    "gdpx.providers.mace.manager",
    "MaceManager",
    {"ase.calculator":"ase","lammps.potential":"lammps"},
    trainer=("gdpx.providers.mace.trainer", "MaceTrainer"),
)
MACE_PROVIDER = add_capabilities(
    MACE_PROVIDER,
    CapabilityKind.DATASET_CODEC,
    {"default": TrainerFactory("gdpx.providers.mace.data", "MaceDataloader")},
)

__all__ = ["MaceManager", "MACE_PROVIDER"]
