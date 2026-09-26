"""Allegro potential provider."""

from ..adapters import manager_provider
from .manager import AllegroManager

ALLEGRO_PROVIDER = manager_provider(
    "allegro", "gdpx.providers.allegro.manager", "AllegroManager",
    {"ase.calculator": ("ase", "lammps"), "lammps.potential": "lammps"},
)

__all__ = ["AllegroManager", "ALLEGRO_PROVIDER"]
