#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
from typing import NoReturn

from GDPy.core.variable import Variable
from GDPy.core.operation import Operation
from GDPy.computation.worker.drive import DriverBasedWorker
from GDPy.scheduler import create_scheduler

class ValidatorNode(Variable):

    def __init__(self, **kwargs):
        """"""
        initial_value = copy.deepcopy(kwargs)
        super().__init__(initial_value)

        return
    

class test(Operation):

    def __init__(self, node_with_structures, validator, potter) -> NoReturn:
        """"""
        input_nodes = [node_with_structures, validator, potter]
        super().__init__(input_nodes)

        return
    
    def forward(self, frames, validator_params, potter):
        """"""
        super().forward()
        # - create a worker
        driver = potter.create_driver()
        scheduler = create_scheduler()
        worker = DriverBasedWorker(potter, driver, scheduler)
        worker.directory = self.directory

        # - create a validator
        method = validator_params.get("method", "minima")
        if method == "singlepoint":
            from GDPy.validator.singlepoint import SinglePointValidator
            rv = SinglePointValidator(self.directory, validator_params, worker)
        else:
            ...

        # - run validation
        rv.run()

        return 

if __name__ == "__main__":
    ...