#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import itertools
import time
from typing import NoReturn, Union, List

from ase import Atoms
from ase.io import read, write

from GDPy.core.operation import Operation
from GDPy.core.register import registers
from GDPy.computation.worker.drive import DriverBasedWorker

@registers.operation.register
class work(Operation):

    def __init__(self, potter, driver, scheduler) -> NoReturn:
        """"""
        super().__init__([potter,driver,scheduler])

        return
    
    def forward(self, potter, drivers: List[dict], scheduler) -> List[DriverBasedWorker]:
        """"""
        super().forward()

        # - create workers
        # TODO: broadcast potters, schedulers as well?
        workers = []
        for i, driver_params in enumerate(drivers):
            # workers share calculator in potter
            driver = potter.create_driver(driver_params)
            worker = DriverBasedWorker(potter, driver, scheduler)
            # wdir is temporary as it may be reset by drive operation
            worker.directory = self.directory / f"w{i}"
            workers.append(worker)

        return workers

@registers.operation.register
class drive(Operation):

    """Drive structures.
    """

    def __init__(
        self, builder, worker, batchsize: int=None,
    ):
        """"""
        super().__init__([builder, worker])

        self.batchsize = batchsize

        return
    
    def forward(self, frames, workers: List[DriverBasedWorker]) -> List[DriverBasedWorker]:
        """Run simulations with given structures and workers.

        Workers' working directory and batchsize are probably set.

        Returns:
            Workers with correct directory and batchsize.

        """
        super().forward()

        # - basic input candidates
        nframes = len(frames)

        # - create workers
        for i, worker in enumerate(workers):
            worker.directory = self.directory / f"w{i}"
            if self.batchsize is not None:
                worker.batchsize = self.batchsize
            else:
                worker.batchsize = nframes
        nworkers = len(workers)

        if nworkers == 1:
            workers[0].directory = self.directory

        # - run workers
        for i, worker in enumerate(workers):
            flag_fpath = worker.directory/"FINISHED"
            self.pfunc(f"run worker {i} for {nframes} nframes")
            if not flag_fpath.exists():
                worker.run(frames)
                worker.inspect(resubmit=True)
                # TODO: check whether finished this operation...
                #       set flag to worker?
                if worker.get_number_of_running_jobs() == 0:
                    with open(flag_fpath, "w") as fopen:
                        fopen.write(
                            f"FINISHED AT {time.asctime( time.localtime(time.time()) )}."
                        )
            else:
                with open(flag_fpath, "r") as fopen:
                    content = fopen.readlines()
                self.pfunc(content)

        return workers

@registers.operation.register
class extract(Operation):

    """Extract dynamics trajectories from a drive-node's worker.
    """

    def __init__(
        self, drive: drive, 
        reduce_cand: bool=True, reduce_work: bool=False,
        read_traj=True, traj_period: int=1,
        include_first=True, include_last=True
    ) -> NoReturn:
        """"""
        super().__init__([drive])

        self.reduce_cand = reduce_cand
        self.reduce_work = reduce_work

        self.read_traj = read_traj
        self.traj_period = traj_period

        self.include_first = include_first
        self.include_last = include_last

        return
    
    def forward(self, workers: List[DriverBasedWorker]):
        """
        Args:
            workers: ...
        
        Returns:
            reduce_cand is false and reduce_work is false -> List[List[List[Atoms]]]
            reduce_cand is true  and reduce_work is false -> List[List[Atoms]]
            reduce_cand is false and reduce_work is true  -> List[List[Atoms]]
            reduce_cand is true  and reduce_work is true  -> List[Atoms]
            
        """
        super().forward()

        self.pfunc(f"reduce_cand {self.reduce_cand} reduce_work {self.reduce_work}")
        
        # TODO: reconstruct trajs to List[List[Atoms]]
        self.workers = workers # for operations to access
        nworkrers = len(workers)

        is_finished = True

        trajectories = [] # List[List[List[Atoms]]], worker->candidate->trajectory
        for i, worker in enumerate(workers):
            # TODO: How to save trajectories into one file?
            #       probably use override function for read/write
            cached_trajs_fpath = self.directory/f"{worker.directory.name}-w{i}.xyz"
            if not cached_trajs_fpath.exists():
                if not (worker.get_number_of_running_jobs() == 0):
                    self.pfunc(f"{worker.directory.name} is not finished.")
                    is_finished = False
                    exit()
                    break
                curr_trajectories_ = worker.retrieve(
                    read_traj = self.read_traj, traj_period = self.traj_period,
                    include_first = self.include_first, include_last = self.include_last,
                    ignore_retrieved=False # TODO: ignore misleads...
                ) # List[List[Atoms]]
                if self.reduce_cand:
                    curr_trajectories = list(itertools.chain(*curr_trajectories_))
                else:
                    curr_trajectories = curr_trajectories_
                write(cached_trajs_fpath, curr_trajectories)
            else:
                curr_trajectories = read(cached_trajs_fpath, ":")
            ntrajs = len(curr_trajectories)
            self.pfunc(f"worker_{i} ntrajectories {ntrajs}")

            trajectories.append(curr_trajectories)

        structures = []
        if self.reduce_work:
            structures = list(itertools.chain(*trajectories))
        else:
            structures = trajectories
        nstructures = len(structures)
        self.pfunc(f"nstructures {nstructures}")

        return structures


if __name__ == "__main__":
    ...
