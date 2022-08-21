#!/usr/bin/env python3
# -*- coding: utf-8 -*

import copy


""" create machine
"""

def create_machine(params_):
    """
    """
    params = copy.deepcopy(params_) # NOTE: params_ may be used many times
    if params is None:
        from GDPy.machine.machine import LocalMachine
        machine = LocalMachine()
    else:
        backend = params.pop("backend", None)
        if backend == "slurm":
            from GDPy.machine.machine import SlurmMachine
            machine = SlurmMachine(**params)
        elif backend == "pbs":
            from GDPy.machine.machine import PbsMachine
            pass
        else:
            pass

    return machine