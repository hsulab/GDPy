#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
import itertools
import pathlib
import time
from typing import Optional, List, Tuple, Union

import omegaconf

import numpy as np

from ase import Atoms
from ase.io import read, write

from .. import config
from ..core.operation import Operation
from ..core.register import registers
from ..core.variable import Variable
from ..data.array import AtomsNDArray
from ..selector.scf import ScfSelector
from ..utils.command import CustomTimer
from ..worker.drive import (
    CommandDriverBasedWorker,
    DriverBasedWorker,
    QueueDriverBasedWorker,
)
from ..worker.interface import ComputerVariable

# --- variable ---


@registers.variable.register
class DriverVariable(Variable):

    def __init__(self, **kwargs):
        """"""
        # - compat
        copied_params = copy.deepcopy(kwargs)
        merged_params = dict(
            task=copied_params.get("task", "min"),
            backend=copied_params.get("backend", "external"),
            ignore_convergence=copied_params.get("ignore_convergence", False),
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
            else:  # int, float, string
                n = 1
            plengths.append((k, n))
        plengths = sorted(plengths, key=lambda x: x[1])
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
    directory: pathlib.Path,
    workers: List[DriverBasedWorker],
    *,
    safe_inspect: bool = True,
    use_archive: bool = True,
    print_func=print,
    debug_func=print,
) -> Tuple[str, AtomsNDArray]:
    """"""
    _print = print_func
    _debug = debug_func

    if not directory.exists():
        directory.mkdir(parents=True, exist_ok=True)

    nworkers = len(workers)
    worker_status = [False] * nworkers

    _debug(f"workers: {workers}")

    trajectories = []
    for i, worker in enumerate(workers):
        # TODO: How to save trajectories into one file?
        #       probably use override function for read/write
        #       i - worker, j - cand
        _print(f"worker: {str(worker.directory)}")
        cached_trajs_dpath = directory / f"{worker.directory.parent.name}-w{i}"
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
            AtomsNDArray(curr_trajectories).save_file(cached_trajs_dpath / "dataset.h5")
        else:
            curr_trajectories = AtomsNDArray.from_file(
                cached_trajs_dpath / "dataset.h5"
            ).tolist()

        trajectories.append(curr_trajectories)

        worker_status[i] = True

    trajectories = AtomsNDArray(trajectories)

    status = "unfinished"
    if all(worker_status):
        status = "finished"
    else:
        ...

    return status, trajectories


def get_shape_data(shape_dir: Union[str, pathlib.Path]):
    # - load previous structures' shape
    if shape_dir.exists():
        inp_shape = np.loadtxt(shape_dir / "shape.dat", dtype=int)
        inp_markers = np.loadtxt(shape_dir / "markers.dat", dtype=int)
    else:
        inp_shape = None
        inp_markers = None

    return inp_shape, inp_markers


def convert_results_to_structures(
    structures: List[Atoms],
    inp_shape,
    inp_markers,
    *,
    reduce_single_worker: bool = True,
    merge_workers: bool = False,
    print_func=print,
    debug_func=print,
):
    """Convert `compute` resulst into structures with a proper shape."""
    # -
    _print = print_func
    _debug = debug_func

    # TODO: Convert to correct input data shape for spc workers...
    #       Optimise the codes here?
    _print(f"target structure shape: {inp_shape}")
    _print(f"input  structure shape: {structures.shape}")
    if inp_shape is not None and inp_markers is not None:
        converted_structures = []
        for curr_structures in structures:  # shape (nworkers, ncandidates, 1)
            curr_structures = list(itertools.chain(*curr_structures))
            inp_shape_ = np.max(inp_markers, axis=0) + 1
            # assert np.allclose(
            #    inp_shape_, inp_shape
            # ), "Inconsistent shape {inp_shape_} vs. {inp_shape}"
            _print(f"previous  structure shape: {inp_shape}")
            _print(f"target    structure shape: {inp_shape_}")
            # - get a full list of indices and fill None to a flatten Atoms List
            # _print(inp_markers)
            curr_converted_structures = []
            full_list = list(itertools.product(*[range(x) for x in inp_shape_]))
            for i, iloc in enumerate(full_list):
                if iloc in inp_markers:
                    curr_converted_structures.append(
                        curr_structures[inp_markers.index(iloc)]
                    )
                else:
                    curr_converted_structures.append(None)
            # _print(len(curr_converted_structures))
            # _print(curr_converted_structures[0])
            # - reshape
            for s in inp_shape_[:0:-1]:
                npoints = len(curr_converted_structures)
                repeats = int(npoints / s)
                reshaped_converted_structures = [
                    [curr_converted_structures[i] for i in range(r * s, (r + 1) * s)]
                    for r in range(repeats)
                ]
                curr_converted_structures = reshaped_converted_structures
            converted_structures.append(curr_converted_structures)
        converted_structures = AtomsNDArray(converted_structures)
    else:  # No data available to convert structures
        converted_structures = structures

    # --- further convert
    if reduce_single_worker:  #  nworkers == 1
        converted_structures = converted_structures[0]

    if merge_workers:  # squeeze the dimension one...
        converted_structures = list(itertools.chain(*converted_structures))
    converted_structures = AtomsNDArray(converted_structures)
    _print(f"extracted_results: {converted_structures}")

    return converted_structures


@registers.operation.register
class compute(Operation):
    """Drive structures."""

    def __init__(
        self,
        builder: Variable,
        worker: Variable,
        batchsize: Optional[int] = None,
        share_wdir: bool = False,
        retain_info: bool = False,
        extract_data: bool = True,
        use_archive: bool = True,
        merge_workers: bool = False,
        reduce_single_worker: bool = True,
        check_scf_convergence: bool = False,
        directory: Union[str, pathlib.Path] = "./",
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
        self.reduce_single_worker = reduce_single_worker

        # -
        self.check_scf_convergence = check_scf_convergence

        return

    def _preprocess_input_nodes(self, input_nodes):
        """"""
        builder, worker = input_nodes
        if isinstance(worker, dict) or isinstance(
            worker, omegaconf.dictconfig.DictConfig
        ):
            worker = ComputerVariable(**worker)

        return builder, worker

    def forward(
        self,
        structures: Union[List[Atoms], AtomsNDArray],
        workers: List[DriverBasedWorker],
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

        nworkers = len(workers)

        # TODO: It is better to move this part to driver...
        #       We only convert spc worker structures shape here...
        driver0_dict = workers[0].driver.as_dict()
        if (
            nworkers == 1
            and driver0_dict.get("task", "min") == "min"
            and (driver0_dict.get("steps", 0) <= 0)
        ):  # TODO: spc?
            # - check input data type
            inp_shape, inp_markers = None, None
            if isinstance(structures, AtomsNDArray):
                frames = structures.get_marked_structures()
                nframes = len(frames)
                inp_shape = structures.shape
                inp_markers = [tuple(iloc.tolist()) for iloc in structures.markers]
            else:  # assume it is just a List of Atoms
                frames = structures
                nframes = len(frames)
                inp_shape = (1, nframes)
                inp_markers = [(0, i) for i in range(nframes)]
            # self._print(inp_markers)
            # -- Dump shape data
            # NOTE: Since all workers use the same input structures,
            #       we only need to dump once here
            shape_dir = self.directory / "_shape"
            shape_dir.mkdir(parents=True, exist_ok=True)
            np.savetxt(
                shape_dir / "shape.dat", np.array(inp_shape, dtype=np.int32), fmt="%8d"
            )
            np.savetxt(
                shape_dir / "markers.dat",
                np.array(inp_markers, dtype=np.int32),
                fmt="%8d",
            )
        else:
            if isinstance(structures, AtomsNDArray):
                frames = structures.get_marked_structures()
            else:  # assume it is just a plain List of Atoms
                frames = structures
            nframes = len(frames)
            inp_shape, inp_markers = None, None

        # - create workers
        for i, worker in enumerate(workers):
            worker.directory = self.directory / f"w{i}"
            if self.batchsize is not None:
                worker.batchsize = self.batchsize
            else:
                worker.batchsize = nframes
            # if self.share_wdir and worker.scheduler.name == "local":
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
            flag_fpath = worker.directory / "FINISHED"
            self._print(f"run worker {i} for {nframes} nframes")
            if not flag_fpath.exists():
                worker.run(frames)
                worker.inspect(resubmit=True)  # if not running, resubmit
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
                status, computed_structures = extract_results_from_workers(
                    self.directory / "extracted",
                    workers,
                    safe_inspect=False,
                    use_archive=self.use_archive,
                    print_func=self._print,
                    debug_func=self._debug,
                )
                self.status = status
                if self.status:
                    self._print(f"extracted structures: {computed_structures}")
                    num_workers = len(workers)
                    reduce_single_worker = self.reduce_single_worker
                    if num_workers != 1:
                        reduce_single_worker = False
                    computed_structures = convert_results_to_structures(
                        computed_structures,
                        inp_shape,
                        inp_markers,
                        reduce_single_worker=reduce_single_worker,
                        merge_workers=self.merge_workers,
                        print_func=self._print,
                        debug_func=self._debug,
                    )
                    if self.check_scf_convergence:
                        selector = ScfSelector(scf_converged=True)
                        selector.directory = self.directory / "selected"
                        selector.select(
                            computed_structures,
                        )
                    output = computed_structures
                else:
                    ...
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

    def __init__(
        self, compute, cache_wdirs: List[Union[str, pathlib.Path]], directory="./"
    ) -> None:
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
        assert (
            nwdirs == nworkers
        ) or nworkers == 1, "Found inconsistent number of cache dirs and workers."

        # - use driver to read results
        cache_data = self.directory / "cache_data.h5"
        if not cache_data.exists():
            from joblib import Parallel, delayed

            # TODO: whether check convergence?
            trajectories = Parallel(n_jobs=config.NJOBS)(
                delayed(self._read_trajectory)(curr_wdir, curr_worker)
                for curr_wdir, curr_worker in itertools.zip_longest(
                    self.cache_wdirs, workers, fillvalue=workers[0]
                )
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
    """Extract dynamics trajectories from a drive-node's worker."""

    def __init__(
        self,
        compute,
        merge_workers: bool = False,
        reduce_single_worker: bool = True,
        use_archive: bool = True,
        check_scf_convergence: bool = False,
        directory="./",
        *args,
        **kwargs,
    ) -> None:
        """Init an extract operation.

        Args:
            compute: Any node forwards a List of workers.
            merge_workers: Whether merge results from different workers togather.
            use_archive: Whether archive computation folders after all workers finished.

        """
        super().__init__(input_nodes=[compute], directory=directory)

        self.merge_workers = merge_workers
        self.reduce_single_worker = reduce_single_worker

        self.use_archive = use_archive
        self.check_scf_convergence = check_scf_convergence

        return

    def forward(self, workers: List[DriverBasedWorker]) -> AtomsNDArray:
        """
        Args:
            workers: ...

        Returns:
            AtomsNDArray.

        """
        super().forward()

        self.workers = workers  # for operations to access

        # - shape
        inp_shape, inp_markers = get_shape_data(
            self.input_nodes[0].directory / "_shape"
        )

        # - extract results
        status, computed_structures = extract_results_from_workers(
            self.directory,
            workers,
            use_archive=self.use_archive,
            print_func=self._print,
            debug_func=self._debug,
        )
        if status:
            self._print(f"extracted structures: {computed_structures}")
            num_workers = len(workers)
            reduce_single_worker = self.reduce_single_worker
            if num_workers != 1:
                reduce_single_worker = False
            computed_structures = convert_results_to_structures(
                computed_structures,
                inp_shape,
                inp_markers,
                reduce_single_worker=reduce_single_worker,
                merge_workers=self.merge_workers,
                print_func=self._print,
                debug_func=self._debug,
            )
            if self.check_scf_convergence:
                selector = ScfSelector(scf_converged=True)
                selector.directory = self.directory / "selected"
                selector.select(
                    computed_structures,
                )

        self.status = status

        return computed_structures


if __name__ == "__main__":
    ...
