#!/usr/bin/env python3
# -*- coding: utf-8 -*


import warnings

"""Create scheduler based on parameters

This module includes several schedulers.

Example:

    .. code-block:: python

        >>> from gdpx.scheduler.local import LocalScheduler
        >>> params = dict()
        >>> scheduler = LocalScheduler(**params)

"""

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
except Exception as e:
    warnings.warn("Module {} import failed: {}".format("remote", e), UserWarning)


if __name__ == "__main__":
    ...
