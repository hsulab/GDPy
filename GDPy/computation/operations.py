#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import itertools
from typing import NoReturn, Union, List

from ase import Atoms
from ase.io import read, write

from GDPy.core.operation import Operation
from GDPy.core.register import registers
from GDPy.computation.worker.drive import DriverBasedWorker

@registers.operation.register
class work(Operation):

    def __init__(self, potter, driver, scheduler, batchsize: int=None) -> NoReturn:
        """"""
        super().__init__([potter,driver,scheduler])

        self.batchsize = batchsize

        return
    
    def forward(self, potter, driver_params, scheduler):
        """"""
        super().forward()

        driver_params = driver_params[0] # support multi-workers?

        driver = potter.create_driver(driver_params) # use external backend
        worker = DriverBasedWorker(potter, driver, scheduler)
        worker.directory = self.directory

        return worker

@registers.operation.register
class drive(Operation):

    """Drive structures.
    """

    def __init__(
        self, builder, potter, driver, scheduler, 
        batchsize: int=None, read_traj=True, traj_period: int=1,
    ):
        """"""
        super().__init__([builder, potter, driver, scheduler])

        self.batchsize = batchsize
        self.read_traj = read_traj
        self.traj_period = traj_period

        return
    
    def forward(self, frames, potter, drivers: List[dict], scheduler) -> List[List[Atoms]]:
        """"""
        super().forward()

        # - basic input candidates
        nframes = len(frames)

        # - create workers
        workers = []
        for i, driver_params in enumerate(drivers):
            # workers share calculator in potter
            driver = potter.create_driver(driver_params) # use external backend
            worker = DriverBasedWorker(potter, driver, scheduler)
            worker.directory = self.directory / f"w{i}"

            if self.batchsize is not None:
                worker.batchsize = self.batchsize
            else:
                worker.batchsize = nframes
            
            workers.append(worker)
        nworkers = len(workers)

        if nworkers == 1:
            workers[0].directory = self.directory
        self.workers = workers

        # - run workers
        ret_groups = []
        for i, worker in enumerate(self.workers):
            self.pfunc(f"run worker {i} for {nframes} nframes")
            cached_fpath = worker.directory/"output.xyz"
            if not cached_fpath.exists():
                worker.run(frames)
                worker.inspect(resubmit=True)
                # TODO: check whether finished this operation...
                new_frames = [] # end frame of each traj
                if worker.get_number_of_running_jobs() == 0:
                    trajectories = worker.retrieve(
                        read_traj = self.read_traj, traj_period = self.traj_period,
                        include_first = True, include_last = True,
                        ignore_retrieved=False # TODO: ignore misleads...
                    )
                    for traj in trajectories: # add last frame
                        new_frames.append(traj[-1])
                    write(cached_fpath, new_frames)
            else:
                new_frames = read(cached_fpath, ":")
            ret_groups.append(new_frames)

        return ret_groups

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
    
    def forward(self, frames: List[Atoms]):
        """
        Args:
            frames: Structures from drive-node's output (end frame of each traj).
        
        Returns:
            reduce_cand is false and reduce_work is false -> List[List[List[Atoms]]]
            reduce_cand is true  and reduce_work is false -> List[List[Atoms]]
            reduce_cand is false and reduce_work is true  -> List[List[Atoms]]
            reduce_cand is true  and reduce_work is true  -> List[Atoms]
            
        """
        super().forward()

        self.pfunc(f"reduce_cand {self.reduce_cand} reduce_work {self.reduce_work}")
        
        # TODO: reconstruct trajs to List[List[Atoms]]
        workers = self.input_nodes[0].workers
        nworkrers = len(workers)

        is_finished = True

        trajectories = [] # List[List[List[Atoms]]], worker->candidate->trajectory
        for i, worker in enumerate(workers):
            cached_trajs_fpath = self.directory/f"w{i}.xyz"
            # TODO: How to save trajectories into one file?
            #       probably use override function for read/write
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
