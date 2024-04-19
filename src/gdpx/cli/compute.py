#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import logging
import pathlib
from typing import Optional, Union, List

from ase.io import write

from ..worker.drive import DriverBasedWorker
from ..reactor.reactor import AbstractReactor


DEFAULT_MAIN_DIRNAME = "MyWorker"


def run_worker(
    structure: List[str],
    worker: DriverBasedWorker,
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
    worker.directory = directory

    _ = worker.run(frames, batch=batch)
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

            # AtomsNDArray(ret).save_file(res_dir/"trajs.h5")
        else:
            print("Results have already been retrieved.")

    return


if __name__ == "__main__":
    ...
