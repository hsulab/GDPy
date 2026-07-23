"""Application-level registry and plugin bootstrap."""

from __future__ import annotations

import importlib

from gdpx import config
from gdpx.core.catalog import registers
from gdpx.core.registry import Register


DOMAIN_REGISTRIES = (
    ("bias", "bias"),
    ("builder", "builder"),
    ("colvar", "colvar"),
    ("comparator", "comparator"),
    ("dataloader", "dataloader"),
    ("describer", "describer"),
    ("expedition", "expedition"),
    ("manager", "potential"),
    ("region", "region"),
    ("scheduler", "scheduler"),
    ("selector", "selector"),
    ("trainer", "trainer"),
    ("validator", "validator"),
)

WORKFLOW_MODULES = (
    "builder",
    "comparator",
    "computer",
    "correction",
    "data",
    "dataset",
    "describer",
    "driver",
    "expedition",
    "potential",
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
    for registry_name, module_name in DOMAIN_REGISTRIES:
        try:
            module = importlib.import_module(f"gdpx.{module_name}")
            setattr(registers, registry_name, module.REGISTER)
        except ImportError as error:
            setattr(registers, registry_name, Register(registry_name))
            errors.append((module_name, error))

    from gdpx.worker.registry import WORKER_REGISTRY

    registers.worker = WORKER_REGISTRY
    modules = [f"gdpx.nodes.{name}" for name in WORKFLOW_MODULES]
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
