"""Compatibility facade for the historical global registry catalog.

New code should import a package-owned registry or a typed factory.  Bootstrap
is intentionally outside :mod:`gdpx.core` and imported lazily here only to
preserve the public API.
"""

from .catalog import registers
from .registry import BaseRegister, Register, Registry


def import_all_modules_for_register(custom_module_paths=None, disable_import_info: bool = False) -> None:
    """Compatibility wrapper for :func:`gdpx.bootstrap.bootstrap_registries`."""
    import importlib

    bootstrap = importlib.import_module("gdpx.bootstrap")
    bootstrap.bootstrap_registries(custom_module_paths, disable_import_info=disable_import_info)
