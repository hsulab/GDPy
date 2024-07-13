#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import logging
import pathlib
from typing import List, Optional, Union

from ase.io import write

from .. import config
from ..builder.interface import BuilderVariable
from ..reactor.reactor import AbstractReactor
from ..scheduler.interface import SchedulerVariable
from ..utils.command import parse_input_file
from ..worker.drive import DriverBasedWorker
from ..worker.grid import GridDriverBasedWorker
from ..worker.interface import ComputerVariable, ReactorVariable
from .build import create_builder

DEFAULT_MAIN_DIRNAME = "MyWorker"


def convert_config_to_potter(config):
    """Convert a configuration file or a dict to a potter/reactor.

    This function is only called in the `main.py`.

    """
    if isinstance(config, dict):
        params = config
    else:  # assume it is json or yaml
        params = parse_input_file(input_fpath=config)

    ptype = params.pop("type", "computer")

    # NOTE: compatibility
    potter_params = params.pop("potter", None)
    potential_params = params.pop("potential", None)
    if potter_params is None:
        if potential_params is not None:
            params["potter"] = potential_params
        else:
            raise RuntimeError("Fail to find any potter (potential) definition.")
    else:
        params["potter"] = potter_params

    if ptype == "computer":
        potter = ComputerVariable(**params).value
    elif ptype == "reactor":

        potter = ReactorVariable(
            potter=params["potter"],
            driver=params.get("driver", None),
            scheduler=params.get("scheduler", {}),
            batchsize=params.get("batchsize", 1),
        ).value[0]
    else:
        ...

    return potter


def run_one_worker(structures, worker, directory, batch, spawn, archive):
    """"""
    worker.directory = directory

    _ = worker.run(structures, batch=batch)
    worker.inspect(resubmit=True, batch=batch)
    if not spawn and worker.get_number_of_running_jobs() == 0:
        res_dir = directory / "results"
        if not res_dir.exists():
            res_dir.mkdir(exist_ok=True)

            ret = worker.retrieve(include_retrieved=True, use_archive=archive)
            if not isinstance(worker.driver, AbstractReactor):
                end_frames = [traj[-1] for traj in ret]
                write(res_dir / "end_frames.xyz", end_frames)
                config._print(f"{directory.name} has already been retrieved.")
            else:
                ...
        else:
            config._print(f"{directory.name} has already been retrieved.")

    return


def run_worker(
    structure: List[str],
    workers: List[DriverBasedWorker],
    *,
    batch: Optional[int] = None,
    spawn: bool = False,
    archive: bool = False,
    directory: Union[str, pathlib.Path] = pathlib.Path.cwd() / DEFAULT_MAIN_DIRNAME,
):
    """"""
    # some imported packages change `logging.basicConfig`
    # and accidently add a StreamHandler to logging.root
    # so remove it...
    for h in logging.root.handlers:
        if isinstance(h, logging.StreamHandler) and not isinstance(
            h, logging.FileHandler
        ):
            logging.root.removeHandler(h)

    # set working directory
    directory = pathlib.Path(directory)
    if not directory.exists():
        directory.mkdir()

    # read structures
    frames = []
    for i, s in enumerate(structure):
        builder = create_builder(s)
        builder.directory = directory / "init" / f"s{i}"
        frames.extend(builder.run())

    # - find input frames
    num_workers = len(workers)
    if num_workers == 1:
        run_one_worker(frames, workers[0], directory, batch, spawn, archive)
    else:
        for i, w in enumerate(workers):
            run_one_worker(frames, w, directory / f"w{i}", batch, spawn, archive)

    return


def convert_config_to_grid_components(grid_params: dict):
    """"""
    # config._print(grid_params)
    grid_data = grid_params.get("grid", None)
    assert grid_data is not None

    # scheduler
    scheduler = SchedulerVariable(**grid_params.get("scheduler", {})).value

    structures, potters, drivers = [], [], []
    for data in grid_data:
        builder_params = data.get("builder")
        builder = BuilderVariable(**builder_params).value
        structures.extend(builder.run())

        # FIXME: broadcast worker to structures?
        computer_params = data.get("computer")
        computers = ComputerVariable(**computer_params).value
        num_computers = len(computers)
        assert num_computers == 1
        potters.append(computers[0].potter)
        drivers.append(computers[0].driver)
        # config._print(f"{builder =}")
        # config._print(f"{computer =}")
        ...
    # config._print(f"{structures =}")
    # config._print(f"{potters =}")
    # config._print(f"{drivers =}")

    # aux parameters
    batchsize = grid_params.get("batchsize", 1)

    aux_params = dict(batchsize=batchsize)

    return scheduler, structures, potters, drivers, aux_params


def run_grid_worker(grid_params: dict, batch: Optional[int], spawn, directory):
    """"""
    directory = pathlib.Path(directory)

    if batch is None:  # submit jobs to queue
        scheduler, structures, potters, drivers, aux_params = (
            convert_config_to_grid_components(grid_params)
        )
        worker = GridDriverBasedWorker(
            potters=potters, drivers=drivers, scheduler=scheduler, **aux_params
        )
        worker._submit = False

        # run computations
        worker.driver = None  # FIXME: compat
        run_one_worker(
            structures, worker, directory, batch=batch, spawn=spawn, archive=True
        )
    else:  # run jobs in command line
        batch_grid_params, batch_wdirs = [], []
        for x in grid_params["grid"]:
            if x["batch"] == batch:
                batch_grid_params.append(x)
                batch_wdirs.append(directory / x["wdir_name"])
        grid_params = {}
        grid_params["grid"] = batch_grid_params

        scheduler, structures, potters, drivers, aux_params = (
            convert_config_to_grid_components(grid_params)
        )
        # aux_params["batchsize"] = len(structures)
        # worker = GridDriverBasedWorker(
        #     potters=potters, drivers=drivers, scheduler=scheduler, **aux_params
        # )
        GridDriverBasedWorker.run_grid_computations_in_command(
            batch_wdirs, structures, drivers, print_func=config._print
        )

    return


if __name__ == "__main__":
    ...
