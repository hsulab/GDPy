#!/usr/bin/env python3
# -*- coding: utf-8 -*


""" create machine
"""

def create_machine(params):
    """
    """
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