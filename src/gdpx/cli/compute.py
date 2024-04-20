#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import logging
import pathlib
from typing import Optional, Union, List

from ase.io import write

from ..worker.drive import DriverBasedWorker
from ..worker.interface import ComputerVariable, ReactorVariable
from ..reactor.reactor import AbstractReactor
from ..utils.command import parse_input_file

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
    worker.inspect(resubmit=True)
    if not spawn and worker.get_number_of_running_jobs() == 0:
        res_dir = directory / "results"
        if not res_dir.exists():
            res_dir.mkdir(exist_ok=True)

            ret = worker.retrieve(include_retrieved=True, use_archive=archive)
            if not isinstance(worker.driver, AbstractReactor):
                end_frames = [traj[-1] for traj in ret]
                write(res_dir / "end_frames.xyz", end_frames)
            else:
                ...
        else:
            print(f"{directory.name} has already been retrieved.")

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

    # - read structures
    from gdpx.builder import create_builder

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
            run_one_worker(frames, w, directory/f"w{i}", batch, spawn, archive)

    return


if __name__ == "__main__":
    ...
