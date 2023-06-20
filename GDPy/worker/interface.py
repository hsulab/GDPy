#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
import itertools
import pathlib
from typing import NoReturn, List

import numpy as np

from ase import Atoms
from ase.io import read, write
from ase.geometry import find_mic

from GDPy.core.operation import Operation
from GDPy.core.variable import Variable
from GDPy.core.register import registers

from .worker import AbstractWorker
from .drive import (
    DriverBasedWorker, CommandDriverBasedWorker, QueueDriverBasedWorker
)
from .single import SingleWorker

DEFAULT_MAIN_DIRNAME = "MyWorker"


@registers.variable.register
class WorkerVariable(Variable):

    """Create a single worker from a dict.
    """

    def __init__(self, potter, driver, scheduler={}, batchsize=1, use_single=False, directory="./", *args, **kwargs):
        """"""
        worker = self._create_worker(potter, driver, scheduler, batchsize, use_single)
        super().__init__(initial_value=worker, directory=directory)

        return
    
    def _create_worker(self, potter_params, driver_params, scheduler_params={}, batchsize=1, use_single=False):
        """"""
        # - potter
        potter_params = copy.deepcopy(potter_params)
        name = potter_params.get("name", None)
        potter = registers.create(
            "manager", name, convert_name=True,
        )
        potter.register_calculator(potter_params.get("params", {}))
        potter.version = potter_params.get("version", "unknown")

        # - driver
        if potter.calc:
            driver_params = copy.deepcopy(driver_params)
            driver = potter.create_driver(driver_params) # use external backend

        # - scheduler
        # default is local machine
        scheduler_params = copy.deepcopy(scheduler_params)
        backend = scheduler_params.pop("backend", "local")
        scheduler = registers.create(
            "scheduler", backend, convert_name=True, **scheduler_params
        )

        if driver and scheduler:
            if not use_single:
                if scheduler.name == "local":
                    from GDPy.worker.drive import CommandDriverBasedWorker as Worker
                else:
                    from GDPy.worker.drive import QueueDriverBasedWorker as Worker
            else:
                from .single import SingleWorker as Worker

            run_worker = Worker(potter, driver, scheduler)
        
        run_worker.batchsize = batchsize

        return run_worker

@registers.variable.register
class ComputerVariable(Variable):

    def __init__(self, potter, driver, scheduler, custom_wdirs=None, use_single=False, *args, **kwargs):
        """"""
        workers = self._create_workers(
            potter.value, driver.value, scheduler.value, 
            custom_wdirs, use_single=use_single
        )
        super().__init__(workers)

        # - save state by all nodes
        self.potter = potter
        self.driver = driver
        self.scheduler = scheduler

        self.custom_wdirs = None
        self.use_single = use_single

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
    
    def _create_workers(self, potter, drivers, scheduler, custom_wdirs=None, use_single=False):
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
            if not use_single:
                if scheduler.name == "local":
                    worker = CommandDriverBasedWorker(potter, driver, scheduler)
                else:
                    worker = QueueDriverBasedWorker(potter, driver, scheduler)
            else:
                worker = SingleWorker(potter, driver, scheduler)
            # wdir is temporary as it may be reset by drive operation
            worker.directory = wdir
            workers.append(worker)
        
        return workers


def run_worker(
    structure: str, directory=pathlib.Path.cwd()/DEFAULT_MAIN_DIRNAME,
    worker: DriverBasedWorker=None, output: str=None, batch: int=None
):
    """"""
    directory = pathlib.Path(directory)
    if not directory.exists():
        directory.mkdir()

    # - read structures
    from GDPy.builder import create_builder
    builder = create_builder(structure)
    builder.directory = directory/"init"

    # - find input frames
    worker.directory = directory

    _ = worker.run(builder, batch=batch)
    worker.inspect(resubmit=True)
    if worker.get_number_of_running_jobs() == 0:
        # - report
        res_dir = directory/"results"
        if not res_dir.exists():
            res_dir.mkdir(exist_ok=True)

            ret = worker.retrieve()
            ret.save_file(res_dir/"trajs.h5")

            end_frames = [traj[-1] for traj in ret]
            write(res_dir/"end_frames.xyz", end_frames)
        else:
            print("Results have already been retrieved.")

    return

if __name__ == "__main__":
    ...
