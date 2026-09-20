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


@dataclass(frozen=True)
class LocalTransportFactory:
    def create(self, parameters: Mapping[str, Any], **context: Any):
        if parameters:
            raise ValueError("Local transport does not accept parameters.")
        scheduler = context.get("scheduler")
        if scheduler is None:
            raise RuntimeError("Transport resolution requires a scheduler instance.")
        return scheduler


@dataclass(frozen=True)
class SshTransportFactory:
    def create(self, parameters: Mapping[str, Any], **context: Any):
        scheduler = context.get("scheduler")
        if scheduler is None:
            raise RuntimeError("Transport resolution requires a scheduler instance.")
        try:
            remote_module = importlib.import_module("gdpx.execution.schedulers.remote")
        except ImportError as error:
            if error.name == "paramiko":
                raise ImportError(
                    "SSH transports require Paramiko; install GDPy with `pip install gdpx[remote]`."
                ) from error
            raise
        transport_class = getattr(remote_module, "SshTransport")
        return transport_class(scheduler=scheduler, **copy.deepcopy(dict(parameters)))


def scheduler_providers():
    definitions = {
        "direct": ("gdpx.execution.schedulers.direct", "DirectScheduler"),
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
    return tuple(providers)


def transport_providers():
    return (
        Provider(
            name="local",
            version="1",
            capabilities={CapabilityKind.TRANSPORT: {"default": LocalTransportFactory()}},
        ),
        Provider(
            name="ssh",
            version="1",
            capabilities={CapabilityKind.TRANSPORT: {"default": SshTransportFactory()}},
        ),
    )
