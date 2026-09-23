#!/usr/bin/env python3
# -*- coding: utf-8 -*


"""Create scheduler based on parameters

This module includes several schedulers.

Example:

    .. code-block:: python

        >>> from gdpx.scheduler.local import LocalScheduler
        >>> params = dict()
        >>> scheduler = LocalScheduler(**params)

"""

from gdpx import config
from gdpx.core.register import BaseRegister

REGISTER = BaseRegister("scheduler")

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


if __name__ == "__main__":
    ...
