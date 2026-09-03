"""Compatibility exports for execution schedulers."""

from importlib import import_module

__all__ = ["BaseScheduler", "LocalScheduler", "LsfScheduler", "PbsScheduler", "SlurmScheduler"]

_EXPORTS = {
    "BaseScheduler": ("gdpx.scheduler.scheduler", "BaseScheduler"),
    "LocalScheduler": ("gdpx.scheduler.local", "LocalScheduler"),
    "LsfScheduler": ("gdpx.scheduler.lsf", "LsfScheduler"),
    "PbsScheduler": ("gdpx.scheduler.pbs", "PbsScheduler"),
    "SlurmScheduler": ("gdpx.scheduler.slurm", "SlurmScheduler"),
}


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module_name, attribute = _EXPORTS[name]
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value

