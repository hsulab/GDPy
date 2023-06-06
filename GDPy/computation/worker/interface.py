#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
import pathlib
from typing import NoReturn, List

import numpy as np

from ase import Atoms
from ase.io import read, write
from ase.geometry import find_mic

from GDPy.core.operation import Operation
from GDPy.core.variable import Variable
from GDPy.core.register import registers
from GDPy.computation.worker.worker import AbstractWorker
from GDPy.computation.worker.drive import DriverBasedWorker

from GDPy.scheduler.interface import create_scheduler

DEFAULT_MAIN_DIRNAME = "MyWorker"


@registers.variable.register
class WorkerVariable(Variable):

    """Create a single worker from a dict.
    """

    def __init__(self, potter, driver, scheduler={}, batchsize=1, directory="./", *args, **kwargs):
        """"""
        worker = self._create_worker(potter, driver, scheduler, batchsize)
        super().__init__(initial_value=worker, directory=directory)

        return
    
    def _create_worker(self, potter_params, driver_params, scheduler_params={}, batchsize=1):
        """"""
        name = potter_params.get("name", None)
        potter = registers.create(
            "manager", name, convert_name=True,
        )
        potter.register_calculator(potter_params.get("params", {}))
        potter.version = potter_params.get("version", "unknown")

        if potter.calc:
            driver = potter.create_driver(driver_params) # use external backend

        # default is local machine
        scheduler = create_scheduler(scheduler_params)

        if driver and scheduler:
            if scheduler.name == "local":
                from GDPy.computation.worker.drive import CommandDriverBasedWorker as Worker
                run_worker = Worker(potter, driver, scheduler)
            else:
                from GDPy.computation.worker.drive import QueueDriverBasedWorker as Worker
                run_worker = Worker(potter, driver, scheduler)
        
        run_worker.batchsize = batchsize

        return run_worker


def run_worker(
    structure: str, directory=pathlib.Path.cwd()/DEFAULT_MAIN_DIRNAME,
    worker: DriverBasedWorker=None, output: str=None, batch: int=None
):
    """"""
    directory = pathlib.Path(directory)
    if not directory.exists():
        directory.mkdir()

    # - read structures
    from GDPy.builder import create_generator
    generator = create_generator(structure)
    generator.directory = directory/"init"

    # - find input frames
    worker.directory = directory

    _ = worker.run(generator, batch=batch)
    worker.inspect(resubmit=True)
    if worker.get_number_of_running_jobs() == 0:
        # - report
        res_dir = directory/"results"
        res_dir.mkdir(exist_ok=True)

        ret = worker.retrieve()
        ret.save_file(res_dir/"trajs.h5")

        end_frames = [traj[-1] for traj in ret]
        write(res_dir/"end_frames.xyz", end_frames)

    return

if __name__ == "__main__":
    ...
