"""Typed, session-independent construction API."""

from importlib import import_module


__all__ = [
    "create_builder",
    "create_region",
    "create_dataloader",
    "create_scheduler",
    "create_validator",
    "create_workers",
    "create_worker_chains",
    "create_selector",
    "create_comparator",
    "create_describer",
    "create_trainer",
    "create_expedition",
]

_EXPORTS = {
    "create_builder": ("gdpx.structures.builders.factory", "canonicalise_builder"),
    "create_region": ("gdpx.structures.regions.factory", "create_region"),
    "create_dataloader": ("gdpx.data.loaders.factory", "create_dataloader"),
    "create_scheduler": ("gdpx.execution.schedulers.factory", "canonicalise_scheduler"),
    "create_validator": ("gdpx.analysis.validators.factory", "canonicalise_validator"),
    "create_workers": ("gdpx.execution.factory", "create_workers"),
    "create_worker_chains": ("gdpx.execution.factory", "create_worker_chains"),
    "create_selector": ("gdpx.workflow.factory", "create_selector"),
    "create_comparator": ("gdpx.workflow.factory", "create_comparator"),
    "create_describer": ("gdpx.workflow.factory", "create_describer"),
    "create_trainer": ("gdpx.workflow.factory", "create_trainer"),
    "create_expedition": ("gdpx.workflow.factory", "create_expedition"),
}


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module_name, attribute = _EXPORTS[name]
    value = getattr(import_module(module_name, __name__) if module_name.startswith(".") else import_module(module_name), attribute)
    globals()[name] = value
    return value
