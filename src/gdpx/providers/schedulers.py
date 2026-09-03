"""Execution-infrastructure providers for built-in schedulers."""

import copy
import importlib
from dataclasses import dataclass
from typing import Any, Mapping

from .capabilities import CapabilityKind
from .provider import Provider


@dataclass(frozen=True)
class SchedulerFactory:
    module: str
    class_name: str

    def create(self, parameters: Mapping[str, Any], **context: Any):
        scheduler_class = getattr(importlib.import_module(self.module), self.class_name)
        return scheduler_class(**copy.deepcopy(dict(parameters)))


def scheduler_providers():
    definitions = {
        "local": ("gdpx.execution.schedulers.local", "LocalScheduler"),
        "lsf": ("gdpx.execution.schedulers.lsf", "LsfScheduler"),
        "pbs": ("gdpx.execution.schedulers.pbs", "PbsScheduler"),
        "slurm": ("gdpx.execution.schedulers.slurm", "SlurmScheduler"),
    }
    return tuple(
        Provider(
            name=name,
            version="1",
            capabilities={
                CapabilityKind.SCHEDULER: {"default": SchedulerFactory(module, class_name)}
            },
        )
        for name, (module, class_name) in definitions.items()
    )
