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

from .. import config
from ..core.register import registers

from .local import LocalScheduler

registers.scheduler.register(LocalScheduler)

from .lsf import LsfScheduler

registers.scheduler.register(LsfScheduler)

from .pbs import PbsScheduler

registers.scheduler.register(PbsScheduler)

from .slurm import SlurmScheduler

registers.scheduler.register(SlurmScheduler)

try:
    from .remote import RemoteSlurmScheduler
    registers.scheduler.register(RemoteSlurmScheduler)
except ImportError as e:
    config._print(f"  {'Scheduler':<16s} {'`remote`':<16s} -> require `{e.name}`.")


if __name__ == "__main__":
    ...
