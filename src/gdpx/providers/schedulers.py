"""Execution-infrastructure providers for built-in schedulers."""

import copy
import importlib
from dataclasses import dataclass
from typing import Any, Mapping

from .capabilities import CapabilityKind
from .configuration import ComponentConfig
from .provider import Provider
from .specs import thaw


@dataclass(frozen=True)
class SchedulerFactory:
    module: str
    class_name: str

    def create(self, parameters: Mapping[str, Any], **context: Any):
        scheduler_class = getattr(importlib.import_module(self.module), self.class_name)
        return scheduler_class(**copy.deepcopy(dict(parameters)))


@dataclass(frozen=True)
class RemoteSchedulerFactory:
    def create(self, parameters: Mapping[str, Any], **context: Any):
        data = copy.deepcopy(dict(parameters))
        try:
            nested = data.pop("scheduler")
        except KeyError as error:
            raise ValueError("Remote scheduler parameters require a nested `scheduler` component.") from error
        if not isinstance(nested, Mapping):
            raise TypeError("Nested remote scheduler configuration must be a mapping.")
        nested_config = ComponentConfig(**dict(nested))
        if nested_config.provider in {"local", "remote"}:
            raise ValueError("Remote scheduler must wrap a non-local queue scheduler.")
        providers = context.get("providers")
        if providers is None:
            raise RuntimeError("Remote scheduler resolution requires the active provider manager.")
        factory = providers.require(
            nested_config.provider,
            CapabilityKind.SCHEDULER,
            nested_config.method or "default",
        )
        scheduler = factory.create(thaw(nested_config.parameters), providers=providers)
        try:
            remote_module = importlib.import_module("gdpx.execution.schedulers.remote")
        except ImportError as error:
            if error.name == "paramiko":
                raise ImportError(
                    "Remote schedulers require Paramiko; install GDPy with `pip install gdpx[remote]`."
                ) from error
            raise
        remote_class = getattr(remote_module, "RemoteScheduler")
        return remote_class(scheduler=scheduler, **data)


def scheduler_providers():
    definitions = {
        "local": ("gdpx.execution.schedulers.local", "LocalScheduler"),
        "lsf": ("gdpx.execution.schedulers.lsf", "LsfScheduler"),
        "pbs": ("gdpx.execution.schedulers.pbs", "PbsScheduler"),
        "slurm": ("gdpx.execution.schedulers.slurm", "SlurmScheduler"),
    }
    providers = [
        Provider(
            name=name,
            version="1",
            capabilities={
                CapabilityKind.SCHEDULER: {"default": SchedulerFactory(module, class_name)}
            },
        )
        for name, (module, class_name) in definitions.items()
    ]
    providers.append(
        Provider(
            name="remote",
            version="1",
            capabilities={CapabilityKind.SCHEDULER: {"default": RemoteSchedulerFactory()}},
        )
    )
    return tuple(providers)
