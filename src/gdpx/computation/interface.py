#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
import itertools
import pathlib
import time

from typing import NoReturn, Union, List, Tuple

import omegaconf

from ase import Atoms
from ase.io import read, write

from .. import config
from ..core.variable import Variable
from ..core.operation import Operation
from ..core.register import registers

from ..data.array import AtomsNDArray
from ..worker.drive import (
    DriverBasedWorker, CommandDriverBasedWorker, QueueDriverBasedWorker
)
from ..utils.command import CustomTimer
from ..worker.interface import ComputerVariable


# --- variable ---

@registers.variable.register
class DriverVariable(Variable):

    def __init__(self, **kwargs):
        """"""
        # - compat
        copied_params = copy.deepcopy(kwargs)
        merged_params = dict(
            task = copied_params.get("task", "min"),
            backend = copied_params.get("backend", "external"),
            ignore_convergence = copied_params.get("ignore_convergence", False)
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

# --- operation ---

def extract_results_from_workers(
        directory: pathlib.Path, workers: List[DriverBasedWorker], *,
        safe_inspect: bool= True,
        use_archive: bool=True, merge_workers: bool=False, 
        print_func=print, debug_func=print
    ) -> Tuple[str, List[AtomsNDArray]]:
    """"""
    _print = print_func
    _debug = debug_func

    if not directory.exists():
        directory.mkdir(parents=True, exist_ok=True)

    nworkers = len(workers)
    worker_status = [False]*nworkers

    _debug(f"workers: {workers}")

    trajectories = []
    for i, worker in enumerate(workers):
        # TODO: How to save trajectories into one file?
        #       probably use override function for read/write
        #       i - worker, j - cand
        _print(f"worker: {str(worker.directory)}")
        cached_trajs_dpath = directory/f"{worker.directory.parent.name}-w{i}"
        if not cached_trajs_dpath.exists():
            if safe_inspect:
                # inspect again for using extract without drive
                worker.inspect(resubmit=False)
                if not (worker.get_number_of_running_jobs() == 0):
                    _print(f"{worker.directory} is not finished.")
                    break
            else:
                # If compute enables extract, it has already done the inspects 
                # thus we can skip them here.
                ...
            cached_trajs_dpath.mkdir(parents=True, exist_ok=True)
            curr_trajectories = worker.retrieve(
                include_retrieved=True, use_archive=use_archive
            )
            AtomsNDArray(curr_trajectories).save_file(cached_trajs_dpath/"dataset.h5")
        else:
            curr_trajectories = AtomsNDArray.from_file(
                cached_trajs_dpath/"dataset.h5"
            ).tolist()

        trajectories.append(curr_trajectories)

        worker_status[i] = True

    status = "unfinished"
    if all(worker_status):
        if nworkers == 1:
            trajectories = trajectories[0]

        if merge_workers:
            trajectories = list(itertools.chain(*trajectories))
        trajectories = AtomsNDArray(trajectories)
        _print(f"extracted_results: {trajectories}")
        status = "finished"
    else:
        ...

    return status, trajectories

@registers.operation.register
class compute(Operation):

    """Drive structures.
    """

    def __init__(
        self, builder: Variable, worker: Variable, 
        batchsize: int=None, share_wdir: bool=False, retain_info: bool=False, 
        extract_data: bool=True, use_archive: bool=True, merge_workers: bool=False,
        directory="./",
    ):
        """Initialise a compute operation.

        Args:
            builder: A builder node.
            worker: A worker node.
            batchsize: Worker's batchsize can be overwritten by this.
            share_wdir: Worker's share_wdir can be overwritten by this.
            retain_info: Worker's retain_info parameter.
            directory: Operation's directory that stores results can be set by Session.

        """
        super().__init__(input_nodes=[builder, worker], directory=directory)

        # - worker-run-related settings
        self.batchsize = batchsize
        self.share_wdir = share_wdir
        self.retain_info = retain_info

        # - worker-retrieve-related settings
        self.extract_data = extract_data
        self.use_archive = use_archive
        self.merge_workers = merge_workers

        return

    def _preprocess_input_nodes(self, input_nodes):
        """"""
        builder, worker = input_nodes
        if isinstance(worker, dict) or isinstance(worker, omegaconf.dictconfig.DictConfig):
            worker = ComputerVariable(**worker)

        return builder, worker
    
    def forward(
            self, frames, workers: List[DriverBasedWorker]
        ) -> Union[List[DriverBasedWorker], List[AtomsNDArray]]:
        """Run simulations with given structures and workers.

        Workers' working directory and batchsize are probably set.
        Sometimes we need workers as some operations can be applied while
        other times we need directly the trajectories to save operation definitions
        in our session configuration files.

        Returns:
            Workers with correct directory and batchsize or AtomsNDArray.

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
        
        output = workers
        if all(worker_status):
            if self.extract_data:
                self._print("--- extract results ---")
                status, trajectories = extract_results_from_workers(
                    self.directory/"extracted", workers, safe_inspect=False,
                    use_archive=self.use_archive, merge_workers=self.merge_workers, 
                    print_func=self._print, debug_func=self._debug
                )
                self.status = status
                output = trajectories
            else:
                self.status = "finished"
        else:
            ...

        return output

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

    def __init__(
            self, compute, merge_workers=False, use_archive: bool=True, 
            directory="./", *args, **kwargs
        ) -> None:
        """Init an extract operation.

        Args:
            compute: Any node forwards a List of workers.
            merge_workers: Whether merge results from different workers togather.
            use_archive: Whether archive computation folders after all workers finished.
        
        """
        super().__init__(input_nodes=[compute], directory=directory)

        self.merge_workers = merge_workers
        self.use_archive = use_archive

        return
    
    def forward(self, workers: List[DriverBasedWorker]) -> AtomsNDArray:
        """
        Args:
            workers: ...
        
        Returns:
            AtomsNDArray.
            
        """
        super().forward()

        self.workers = workers # for operations to access

        status, trajectories = extract_results_from_workers(
            self.directory, workers, use_archive=self.use_archive,
            merge_workers=self.merge_workers, 
            print_func=self._print, debug_func=self._debug
        )
        self.status = status

        return trajectories


if __name__ == "__main__":
    ...
