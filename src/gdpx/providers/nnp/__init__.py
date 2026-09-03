"""NNP software provider."""

from ..adapters import manager_provider
from .calculator import ACSFNN
from .descriptor import compute_n_features
from .manager import NnAcsfManager

NNP_PROVIDER = manager_provider(
    "nnp", "gdpx.providers.nnp.manager", "NnAcsfManager",
    {"ase.calculator": "ase"},
    trainer=("gdpx.providers.nnp.trainer", "NnpTrainer"),
)

__all__ = ["ACSFNN", "compute_n_features", "NnAcsfManager", "NNP_PROVIDER"]
