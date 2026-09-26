"""Worker public API.

Exports are loaded lazily so importing one worker implementation does not pull
in explorations, trainers, workflow nodes, and every optional backend.
"""

from importlib import import_module


__all__ = [
    "BaseWorker",
    "DriverBasedWorker",
    "DriverBatchError",
    "DriverFailure",
    "SingleWorker",
    "ExplorationBasedWorker",
    "TrainerBasedWorker",
    "ReactorBasedWorker",
    "JobRecord",
    "JobStore",
    "run_computation_in_commandline",
    "run_exploration_in_commandline",
]

_EXPORTS = {
    "BaseWorker": (".worker", "BaseWorker"),
    "DriverBasedWorker": (".drive", "DriverBasedWorker"),
    "DriverBatchError": (".drive", "DriverBatchError"),
    "DriverFailure": (".drive", "DriverFailure"),
    "run_computation_in_commandline": (".drive", "run_computation_in_commandline"),
    "SingleWorker": (".single", "SingleWorker"),
    "ExplorationBasedWorker": (".explore", "ExplorationBasedWorker"),
    "run_exploration_in_commandline": (".explore", "run_exploration_in_commandline"),
    "TrainerBasedWorker": (".train", "TrainerBasedWorker"),
    "ReactorBasedWorker": (".react", "ReactorBasedWorker"),
    "JobRecord": (".store", "JobRecord"),
    "JobStore": (".store", "JobStore"),
}


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module_name, attribute = _EXPORTS[name]
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value
