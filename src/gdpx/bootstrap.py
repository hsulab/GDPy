"""Application-level registry and plugin bootstrap."""

from __future__ import annotations

import importlib

from gdpx import config
WORKFLOW_MODULES = (
    "builder",
    "comparator",
    "correction",
    "data",
    "dataset",
    "describer",
    "driver",
    "expedition",
    "runtime",
    "reactor",
    "region",
    "scheduler",
    "selector",
    "trainer",
    "validator",
)


def bootstrap_registries(custom_module_paths=None, *, disable_import_info: bool = False) -> None:
    """Load domain registries, workers, workflow adapters, and plugins in order."""
    errors = []
    from gdpx.providers import get_provider_manager

    get_provider_manager()
    modules = [f"gdpx.workflow.nodes.{name}" for name in WORKFLOW_MODULES]
    if custom_module_paths:
        modules.extend(custom_module_paths)
    for module_name in modules:
        try:
            importlib.import_module(module_name)
        except ImportError as error:
            errors.append((module_name, error))

    if not disable_import_info:
        config._print("FAILED TO IMPORT OPTIONAL MODULES: ")
        for name, error in errors:
            config._print(f"  {name:<33s} -> require `{error.name}`.")
