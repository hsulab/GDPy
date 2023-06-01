#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
import itertools
import pathlib
from typing import NoReturn, List

from ase import Atoms
from ase.io import read, write

from GDPy.core.variable import Variable
from GDPy.core.operation import Operation
from GDPy.core.register import registers
from GDPy.computation.worker.drive import (
    DriverBasedWorker, CommandDriverBasedWorker, QueueDriverBasedWorker
)

@registers.variable.register
class ComputerVariable(Variable):

    def __init__(self, potter, driver, scheduler, custom_wdirs=None, *args, **kwargs):
        """"""
        workers = self._create_workers(
            potter.value, driver.value, scheduler.value, custom_wdirs
        )
        super().__init__(workers)

        # - save state by all nodes
        self.potter = potter
        self.driver = driver
        self.scheduler = scheduler
        self.custom_wdirs = None

        return
    
    def _update_workers(self, potter_node):
        """"""
        if isinstance(potter_node, Variable):
            potter = potter_node.value
        elif isinstance(potter_node, Operation):
            # TODO: ...
            node = potter_node
            if node.preward():
                node.inputs = [input_node.output for input_node in node.input_nodes]
                node.output = node.forward(*node.inputs)
            else:
                print("wait previous nodes to finish...")
            potter = node.output
        else:
            ...
        print("update manager: ", potter)
        print(potter.calc.model_path)
        workers = self._create_workers(
            potter, self.driver.value, self.scheduler.value,
            custom_wdirs=self.custom_wdirs
        )
        self.value = workers

        return
    
    def _create_workers(self, potter, drivers, scheduler, custom_wdirs=None):
        # - check if there were custom wdirs, and zip longest
        ndrivers = len(drivers)
        if custom_wdirs is not None:
            wdirs = [pathlib.Path(p) for p in custom_wdirs]
        else:
            wdirs = [self.directory/f"w{i}" for i in range(ndrivers)]
        
        nwdirs = len(wdirs)
        assert (nwdirs==ndrivers and ndrivers>1) or (nwdirs>=1 and ndrivers==1), "Invalid wdirs and drivers."
        pairs = itertools.zip_longest(wdirs, drivers, fillvalue=drivers[0])

        # - create workers
        # TODO: broadcast potters, schedulers as well?
        workers = []
        for wdir, driver_params in pairs:
            # workers share calculator in potter
            driver = potter.create_driver(driver_params)
            if scheduler.name == "local":
                worker = CommandDriverBasedWorker(potter, driver, scheduler)
            else:
                worker = QueueDriverBasedWorker(potter, driver, scheduler)
            # wdir is temporary as it may be reset by drive operation
            worker.directory = wdir
            workers.append(worker)
        
        return workers


@registers.variable.register
class DriverVariable(Variable):

    def __init__(self, **kwargs):
        """"""
        # - compat
        copied_params = copy.deepcopy(kwargs)
        merged_params = dict(
            task = copied_params.get("task", "min"),
            backend = copied_params.get("backend", "external"),
        )
        merged_params.update(**copied_params.get("init", {}))
        merged_params.update(**copied_params.get("run", {}))

        initial_value = self._broadcast_drivers(merged_params)

        super().__init__(initial_value)

        return
    
    def _broadcast_drivers(self, params: dict) -> List[dict]:
        """Broadcast parameters if there were any parameter is a list."""
        # - find longest params
        plengths = []
        for k, v in params.items():
            if isinstance(v, list):
                n = len(v)
            else: # int, float, string
                n = 1
            plengths.append((k,n))
        plengths = sorted(plengths, key=lambda x:x[1])
        # NOTE: check only has one list params
        assert sum([p[1] > 1 for p in plengths]) <= 1, "only accept one param as list."

        # - convert to dataclass
        params_list = []
        maxname, maxlength = plengths[-1]
        for i in range(maxlength):
            curr_params = {}
            for k, n in plengths:
                if n > 1:
                    v = params[k][i]
                else:
                    v = params[k]
                curr_params[k] = v
            params_list.append(curr_params)

        return params_list

if __name__ == "__main__":
    ...