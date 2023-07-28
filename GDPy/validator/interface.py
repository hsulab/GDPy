#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
from typing import NoReturn

from ..core.variable import Variable
from ..core.operation import Operation
from ..core.register import registers

from ..worker.drive import DriverBasedWorker
from .validator import AbstractValidator

@registers.variable.register
class ValidatorVariable(Variable):

    def __init__(self, directory="./", **kwargs):
        """"""
        # - create a validator
        method = kwargs.get("method", "minima")
        validator = registers.create("validator", method, convert_name=False, **kwargs)

        # - save
        super().__init__(initial_value=validator, directory=directory)

        return
    
@registers.operation.register
class validate(Operation):

    """The operation to validate properties by potentials.

    The reference properties should be stored and accessed through `structures`.

    """

    def __init__(self, structures, validator, worker, run_params: dict={}, directory="./") -> NoReturn:
        """Init a validate operation.

        Args:
            structures: A node that forwards structures.
            validator: A validator.
            worker: A worker to run calculations. TODO: Make this optional.
        
        """
        input_nodes = [structures, validator, worker]
        super().__init__(input_nodes=input_nodes, directory=directory)

        self.run_params = run_params

        return
    
    def forward(self, dataset, validator: AbstractValidator, workers):
        """Run a validator on input dataset.

        Args:
            dataset: List[(prefix,frames)]

        """
        super().forward()
        # - create a worker
        nworkers = len(workers)
        assert nworkers == 1, "Validator only accepts one worker."
        worker = workers[0]
        worker.directory = self.directory

        # - run validation
        validator.directory = self.directory
        validator.run(dataset, worker, **self.run_params)

        self.status = "finished"

        return # TODO: forward a reference-prediction pair?

if __name__ == "__main__":
    ...
