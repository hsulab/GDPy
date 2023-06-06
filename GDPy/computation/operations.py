#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
import itertools
import pathlib
import time
from typing import NoReturn, Union, List

from ase import Atoms
from ase.io import read, write

from GDPy.core.operation import Operation
from GDPy.core.register import registers
from GDPy.computation.worker.drive import (
    DriverBasedWorker, CommandDriverBasedWorker, QueueDriverBasedWorker
)
from GDPy.data.array import AtomsArray2D
from GDPy.data.trajectory import Trajectories

@registers.operation.register
class work(Operation):

    """Create a list of workers by necessary components (potter, drivers, and scheduler).
    """

    def __init__(self, potter, driver, scheduler, custom_wdirs=None, directory="./", *args, **kwargs) -> NoReturn:
        """"""
        super().__init__(input_nodes=[potter,driver,scheduler], directory=directory)

        # Custom worker directories.
        self.custom_wdirs = custom_wdirs

        return
    
    def forward(self, potter, drivers: List[dict], scheduler) -> List[DriverBasedWorker]:
        """"""
        super().forward()

        # - check if there were custom wdirs, and zip longest
        ndrivers = len(drivers)
        if self.custom_wdirs is not None:
            wdirs = [pathlib.Path(p) for p in self.custom_wdirs]
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
        
        self.status = "finished"

        return workers


@registers.operation.register
class drive(Operation):

    """Drive structures.
    """

    def __init__(
        self, builder, worker, batchsize: int=None, directory="./",
    ):
        """"""
        super().__init__(input_nodes=[builder, worker], directory=directory)

        self.batchsize = batchsize

        return
    
    def forward(self, frames, workers: List[DriverBasedWorker]) -> List[DriverBasedWorker]:
        """Run simulations with given structures and workers.

        Workers' working directory and batchsize are probably set.

        Returns:
            Workers with correct directory and batchsize.

        """
        super().forward()

        if isinstance(frames, AtomsArray2D):
            frames = frames.get_marked_structures()

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
        worker_status = []
        for i, worker in enumerate(workers):
            flag_fpath = worker.directory/"FINISHED"
            self.pfunc(f"run worker {i} for {nframes} nframes")
            if not flag_fpath.exists():
                worker.run(frames)
                worker.inspect(resubmit=True) # if not running, resubmit
                if worker.get_number_of_running_jobs() == 0:
                    with open(flag_fpath, "w") as fopen:
                        fopen.write(
                            f"FINISHED AT {time.asctime( time.localtime(time.time()) )}."
                        )
                    worker_status.append(True)
                else:
                    worker_status.append(False)
            else:
                with open(flag_fpath, "r") as fopen:
                    content = fopen.readlines()
                self.pfunc(content)
                worker_status.append(True)
        
        if all(worker_status):
            self.status = "finished"

        return workers


@registers.operation.register
class extract(Operation):

    """Extract dynamics trajectories from a drive-node's worker.
    """

    def __init__(self, drive: drive, directory="./", *args, **kwargs) -> None:
        """"""
        super().__init__(input_nodes=[drive], directory=directory)

        return
    
    def forward(self, workers: List[DriverBasedWorker]) -> AtomsArray2D:
        """
        Args:
            workers: ...
        
        Returns:
            AtomsArray2D.
            
        """
        super().forward()

        # TODO: reconstruct trajs to List[List[Atoms]]
        self.workers = workers # for operations to access
        nworkers = len(workers)
        self.pfunc(f"nworkers: {nworkers}")
        #print(self.workers)
        worker_status = [False]*nworkers

        trajectories = Trajectories()
        for i, worker in enumerate(workers):
            # TODO: How to save trajectories into one file?
            #       probably use override function for read/write
            #       i - worker, j - cand
            print("worker: ", worker.directory)
            cached_trajs_dpath = self.directory/f"{worker.directory.parent.name}-w{i}"
            if not cached_trajs_dpath.exists():
                # inspect again for using extract without drive
                worker.inspect(resubmit=False)
                if not (worker.get_number_of_running_jobs() == 0):
                    self.pfunc(f"{worker.directory} is not finished.")
                    break
                cached_trajs_dpath.mkdir(parents=True, exist_ok=True)
                curr_trajectories = worker.retrieve(
                    ignore_retrieved=False, # TODO: ignore misleads...
                ) # List[List[Atoms]] or List[Atoms] depends on read_traj
                curr_trajectories.save_file(cached_trajs_dpath/"dataset.h5")
            else:
                #if self.reduce_cand:
                #    curr_trajectories = read(cached_trajs_dpath/"reduced_candidates.xyz", ":")
                #else:
                #    cached_fnames = list(cached_trajs_dpath.glob("cand*"))
                #    cached_fnames.sort()

                #    curr_trajectories = []
                #    for fpath in cached_fnames:
                #        traj = read(fpath, ":")
                #        curr_trajectories.append(traj)
                curr_trajectories = Trajectories.from_file(cached_trajs_dpath/"dataset.h5")

            self.pfunc(f"worker_{i} {curr_trajectories}")
            trajectories.extend(curr_trajectories)

            worker_status[i] = True
        
        structures = []
        if all(worker_status):
            self.pfunc(f"worker_: {trajectories}")
            self.status = "finished"
        else:
            ...

        return trajectories


if __name__ == "__main__":
    ...
