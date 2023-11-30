#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
import itertools
import pathlib
import time
from typing import NoReturn, Union, List

from ase import Atoms
from ase.io import read, write

from .. import config
from gdpx.core.operation import Operation
from gdpx.core.register import registers
from gdpx.worker.drive import (
    DriverBasedWorker, CommandDriverBasedWorker, QueueDriverBasedWorker
)
from gdpx.data.array import AtomsNDArray
from ..utils.command import CustomTimer


@registers.operation.register
class compute(Operation):

    """Drive structures.
    """

    def __init__(
        self, builder, worker, batchsize: int=None, 
        share_wdir: bool=False, retain_info: bool=False, directory="./",
    ):
        """"""
        super().__init__(input_nodes=[builder, worker], directory=directory)

        self.batchsize = batchsize
        self.share_wdir = share_wdir
        self.retain_info = retain_info

        return
    
    def forward(self, frames, workers: List[DriverBasedWorker]) -> List[DriverBasedWorker]:
        """Run simulations with given structures and workers.

        Workers' working directory and batchsize are probably set.

        Returns:
            Workers with correct directory and batchsize.

        """
        super().forward()

        if isinstance(frames, AtomsNDArray):
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
            #if self.share_wdir and worker.scheduler.name == "local":
            if self.share_wdir:
                worker._share_wdir = True
            if self.retain_info:
                worker._retain_info = True
        nworkers = len(workers)

        if nworkers == 1:
            workers[0].directory = self.directory

        # - run workers
        worker_status = []
        for i, worker in enumerate(workers):
            flag_fpath = worker.directory/"FINISHED"
            self._print(f"run worker {i} for {nframes} nframes")
            if not flag_fpath.exists():
                worker.run(frames)
                worker.inspect(resubmit=True) # if not running, resubmit
                if worker.get_number_of_running_jobs() == 0:
                    # -- save flag
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
                self._print(content)
                worker_status.append(True)
        
        if all(worker_status):
            self.status = "finished"

        return workers

@registers.operation.register
class extract_cache(Operation):

    """Extract results from finished (cache) calculation wdirs.

    This is useful when reading results from manually created structures.

    """

    def __init__(self, compute, cache_wdirs: List[Union[str, pathlib.Path]], directory="./") -> None:
        """"""
        super().__init__(input_nodes=[compute], directory=directory)

        self.cache_wdirs = cache_wdirs

        return
    
    @CustomTimer(name="extract_cache", func=config._debug)
    def forward(self, workers: List[DriverBasedWorker]):
        """"""
        super().forward()
        
        # - broadcast workers
        nwdirs = len(self.cache_wdirs)
        nworkers = len(workers)
        assert (nwdirs == nworkers) or nworkers == 1, "Found inconsistent number of cache dirs and workers."

        # - use driver to read results
        cache_data = self.directory/"cache_data.h5"
        if not cache_data.exists():
            from joblib import Parallel, delayed
            # TODO: whether check convergence?
            trajectories = Parallel(n_jobs=config.NJOBS)(
                delayed(self._read_trajectory)(curr_wdir, curr_worker) 
                for curr_wdir, curr_worker in itertools.zip_longest(self.cache_wdirs, workers, fillvalue=workers[0])
            )
            trajectories = AtomsNDArray(data=trajectories)
            trajectories.save_file(cache_data)
        else:
            self._print("read cache...")
            trajectories = AtomsNDArray.from_file(cache_data)

        self.status = "finished"

        return trajectories
    
    @staticmethod
    def _read_trajectory(wdir, worker):
        """"""
        worker.driver.directory = wdir

        return worker.driver.read_trajectory()


@registers.operation.register
class extract(Operation):

    """Extract dynamics trajectories from a drive-node's worker.
    """

    def __init__(self, compute, merge_workers=False, directory="./", *args, **kwargs) -> None:
        """Init an extract operation.

        Args:
            compute: Any node forwards a List of workers.
            merge_workers: Whether merge results from different workers togather.
        
        """
        super().__init__(input_nodes=[compute], directory=directory)

        self.merge_workers = merge_workers

        return
    
    def forward(self, workers: List[DriverBasedWorker]) -> AtomsNDArray:
        """
        Args:
            workers: ...
        
        Returns:
            AtomsNDArray.
            
        """
        super().forward()

        # TODO: reconstruct trajs to List[List[Atoms]]
        self.workers = workers # for operations to access
        nworkers = len(workers)
        self._print(f"nworkers: {nworkers}")
        #print(self.workers)
        worker_status = [False]*nworkers

        trajectories = []
        for i, worker in enumerate(workers):
            # TODO: How to save trajectories into one file?
            #       probably use override function for read/write
            #       i - worker, j - cand
            self._print(f"worker: {str(worker.directory)}")
            cached_trajs_dpath = self.directory/f"{worker.directory.parent.name}-w{i}"
            if not cached_trajs_dpath.exists():
                # inspect again for using extract without drive
                worker.inspect(resubmit=False)
                if not (worker.get_number_of_running_jobs() == 0):
                    self._print(f"{worker.directory} is not finished.")
                    break
                cached_trajs_dpath.mkdir(parents=True, exist_ok=True)
                curr_trajectories = worker.retrieve(
                    include_retrieved=True, 
                )
                AtomsNDArray(curr_trajectories).save_file(cached_trajs_dpath/"dataset.h5")
            else:
                curr_trajectories = AtomsNDArray.from_file(
                    cached_trajs_dpath/"dataset.h5"
                ).tolist()

            trajectories.append(curr_trajectories)

            worker_status[i] = True

        if nworkers == 1:
            trajectories = trajectories[0]

        if self.merge_workers:
            trajectories = list(itertools.chain(*trajectories))
        trajectories = AtomsNDArray(trajectories)
        self._debug(trajectories)
        
        if all(worker_status):
            self._print(f"worker_: {trajectories}")
            self.status = "finished"
        else:
            ...

        return trajectories


if __name__ == "__main__":
    ...
