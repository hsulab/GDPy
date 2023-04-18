#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
from typing import NoReturn

from GDPy.core.variable import Variable
from GDPy.core.operation import Operation
from GDPy.core.register import registers

from GDPy.computation.worker.drive import DriverBasedWorker
from GDPy.scheduler import create_scheduler
from GDPy.validator import AbstractValidator

class ValidatorNode(Variable):

    def __init__(self, **kwargs):
        """"""
        # - create a validator
        method = kwargs.get("method", "minima")
        validator = registers.create("validator", method, self.directory, kwargs)
        print(validator)

        # - save
        initial_value = validator
        super().__init__(initial_value)

        return
    

class test(Operation):

    def __init__(self, node_with_structures, validator, potter) -> NoReturn:
        """"""
        input_nodes = [node_with_structures, validator, potter]
        super().__init__(input_nodes)

        return
    
    def forward(self, frames, validator: AbstractValidator, potter):
        """"""
        super().forward()
        # - create a worker
        driver = potter.create_driver()
        scheduler = create_scheduler()
        worker = DriverBasedWorker(potter, driver, scheduler)
        worker.directory = self.directory

        # - run validation
        validator.directory = self.directory
        validator.worker = worker
        validator.run()

        return 

if __name__ == "__main__":
    ...