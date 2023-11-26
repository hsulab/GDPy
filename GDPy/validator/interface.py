#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
from typing import NoReturn

from ..core.variable import Variable, DummyVariable
from ..core.operation import Operation
from ..core.register import registers

from ..data.array import AtomsNDArray
from ..data.dataset import AbstractDataloader
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

    def __init__(self, structures, validator, worker=DummyVariable(), run_params: dict={}, directory="./") -> None:
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
        if workers is not None:
            nworkers = len(workers)
            assert nworkers == 1, "Validator only accepts one worker."
            worker = workers[0]
            worker.directory = self.directory
        else:
            worker = None

        # NOTE: In an active session, the dataset is dynamic, thus, 
        #       we need load the dataset before run...
        #       Validator accepts dict(reference=[], prediction=[])
        dataset_ = {}
        for k, v in dataset.items():
            if isinstance(v, dict):
                ...
            elif isinstance(v, AtomsNDArray):
                ...
            elif isinstance(v, AbstractDataloader):
                v = v.load_frames()
            else:
                raise RuntimeError(f"{k} Dataset {type(v)} is not a dict or loader.")
            dataset_[k] = v
        dataset = dataset_

        # - run validation
        validator.directory = self.directory
        validator.run(dataset, worker, **self.run_params)

        self.status = "finished"

        return # TODO: forward a reference-prediction pair?
    
    def report_convergence(self, *args, **kwargs) -> bool:
        """"""
        input_nodes = self.input_nodes
        assert hasattr(input_nodes[1], "output"), f"Operation {self.directory.name} cannot report convergence without forwarding."
        validator = input_nodes[1].output

        self._print(f"{validator.__class__.__name__} Convergence")
        if hasattr(validator, "report_convergence"):
            converged = validator.report_convergence()
        else:
            self._print("    >>> True  (No report available)")
            converged = True

        return converged


if __name__ == "__main__":
    ...
