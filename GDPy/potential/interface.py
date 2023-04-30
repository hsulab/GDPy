#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from GDPy.core.variable import Variable
from GDPy.core.register import registers

from GDPy.potential.register import PotentialRegister
from GDPy.potential.register import create_potter


@registers.variable.register
class PotterVariable(Variable):

    def __init__(self, **kwargs):
        """"""
        #manager = PotentialRegister()
        name = kwargs.get("name", None)
        #potter = manager.create_potential(pot_name=name)
        #potter.register_calculator(kwargs.get("params", {}))
        #potter.version = kwargs.get("version", "unknown")

        potter = registers.create(
            "manager", name, convert_name=True, 
            #**kwargs.get("params", {})
        )
        potter.register_calculator(kwargs.get("params", {}))

        super().__init__(potter)

        return

@registers.variable.register
class WorkerVariable(Variable):

    def __init__(self, **kwargs):
        """"""
        worker = create_potter(**kwargs) # DriveWorker or TrainWorker
        super().__init__(worker)

        return

if __name__ == "__main__":
    ...