"""mixer software provider."""

from ..adapters import manager_provider
from .manager import MixerManager

MIXER_PROVIDER = manager_provider(
    "mixer",
    "gdpx.providers.mixer.manager",
    "MixerManager",
    {"ase.calculator":"ase"},
)

__all__ = ["MixerManager", "MIXER_PROVIDER"]

