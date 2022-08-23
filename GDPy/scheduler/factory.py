#!/usr/bin/env python3
# -*- coding: utf-8 -*

import copy


""" create scheduler
"""

def create_scheduler(params_):
    """
    """
    params = copy.deepcopy(params_) # NOTE: params_ may be used many times
    if params is None:
        from GDPy.scheduler.scheduler import LocalScheduler as scheduler_cls
    else:
        backend = params.pop("backend", None)
        if backend == "slurm":
            from GDPy.scheduler.scheduler import SlurmScheduler as scheduler_cls
        elif backend == "pbs":
            from GDPy.scheduler.scheduler import PbsScheduler as scheduler_cls
        else:
            pass
        scheduler = scheduler_cls(**params)

    return scheduler