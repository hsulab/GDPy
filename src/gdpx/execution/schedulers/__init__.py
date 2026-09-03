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

from gdpx import config
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

try:
    from .remote import RemoteSlurmScheduler

    REGISTER.register(RemoteSlurmScheduler)
except ImportError as e:
    config._print(f"  {'Scheduler':<16s} {'`remote`':<16s} -> require `{e.name}`.")


__all__ = ["REGISTER", "BaseScheduler", "LocalScheduler", "LsfScheduler", "PbsScheduler", "SlurmScheduler"]

from .factory import canonicalise_scheduler

__all__.append("canonicalise_scheduler")
