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
    "create_builder": (".builder", "canonicalise_builder"),
    "create_region": (".region", "create_region"),
    "create_dataloader": (".dataloader", "create_dataloader"),
    "create_scheduler": (".scheduler", "canonicalise_scheduler"),
    "create_validator": (".validator", "canonicalise_validator"),
    "create_workers": (".computer", "create_workers"),
    "create_worker_chains": (".computer", "create_worker_chains"),
    "create_selector": (".components", "create_selector"),
    "create_comparator": (".components", "create_comparator"),
    "create_describer": (".components", "create_describer"),
    "create_trainer": (".components", "create_trainer"),
    "create_expedition": (".components", "create_expedition"),
}


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module_name, attribute = _EXPORTS[name]
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value
