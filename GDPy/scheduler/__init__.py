#!/usr/bin/env python3
# -*- coding: utf-8 -*

import copy

from GDPy.scheduler.scheduler import AbstractScheduler


"""Create scheduler based on parameters

This module includes several schedulers.

Example:

    .. code-block:: python

        >>> from GDPy.scheduler import create_scheduler
        >>> params = dict()
        >>> scheduler = create_scheduler(params)

"""

def create_scheduler(params_: dict={}) -> AbstractScheduler:
    """Create a scheduler.
    
    Args:
        params_: Scheduler initialisation parameters.
    
    Returns:
        A scheduler.
    """
    params = copy.deepcopy(params_) # NOTE: params_ may be used many times
    if not params:
        from GDPy.scheduler.local import LocalScheduler as scheduler_cls
    else:
        backend = params.pop("backend", None)
        if backend == "local":
            from GDPy.scheduler.local import LocalScheduler as scheduler_cls
        elif backend == "slurm":
            from GDPy.scheduler.slurm import SlurmScheduler as scheduler_cls
        elif backend == "pbs":
            from GDPy.scheduler.pbs import PbsScheduler as scheduler_cls
        elif backend == "lsf":
            from GDPy.scheduler.lsf import LSFScheduler as scheduler_cls
        else:
            pass
    scheduler = scheduler_cls(**params)

    return scheduler

if __name__ == "__main__":
    pass
