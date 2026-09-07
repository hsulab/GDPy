#!/usr/bin/env python3
# -*- coding: utf-8 -*


"""Create scheduler based on parameters

This module includes several schedulers.

Example:

    .. code-block:: python

        >>> from gdpx.execution.schedulers.local import LocalScheduler
        >>> params = dict()
        >>> scheduler = LocalScheduler(**params)

"""

from gdpx.core.registry import Registry

REGISTER = Registry("scheduler")

from .scheduler import BaseScheduler

from .local import LocalScheduler

REGISTER.register(LocalScheduler)

from .lsf import LsfScheduler

REGISTER.register(LsfScheduler)

from .pbs import PbsScheduler

REGISTER.register(PbsScheduler)

from .slurm import SlurmScheduler

REGISTER.register(SlurmScheduler)

__all__ = [
    "REGISTER",
    "BaseScheduler",
    "LocalScheduler",
    "LsfScheduler",
    "PbsScheduler",
    "RemoteScheduler",
    "SlurmScheduler",
]

from .factory import canonicalise_scheduler

__all__.append("canonicalise_scheduler")


def __getattr__(name):
    if name == "RemoteScheduler":
        from .remote import RemoteScheduler

        return RemoteScheduler
    raise AttributeError(name)
