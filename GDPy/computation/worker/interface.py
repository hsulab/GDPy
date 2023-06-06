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
from GDPy.computation.worker.worker import AbstractWorker
from GDPy.computation.worker.drive import DriverBasedWorker

DEFAULT_MAIN_DIRNAME = "MyWorker"


def run_driver(structure: str, directory="./", worker=None, o_fname=None):
    """"""
    directory = pathlib.Path(directory)
    if not directory.exists():
        directory.mkdir()

    # - read structures
    from GDPy.builder import create_generator
    generator = create_generator(structure)
    generator.directory = directory/"init"
    frames = generator.run()
    #nframes = len(frames)
    #print("nframes: ", nframes)

    wdirs = []
    for i, atoms in enumerate(frames):
        wdir = atoms.info.get("wdir", f"cand{i}")
        wdirs.append(wdir)
    assert len(wdirs) == len(frames), "Have duplicated wdir names..."
    
    driver = worker.driver
    for wdir, atoms in zip(wdirs, frames):
        driver.reset()
        driver.directory = directory/wdir
        print(driver.directory)
        driver.run(atoms, read_exists=True, extra_info=None)
    
    ret_frames = []
    for wdir, atoms in zip(wdirs, frames):
        driver.directory = directory/wdir
        atoms = driver.read_converged()
        ret_frames.append(atoms)
    
    if o_fname is not None:
        write(directory/o_fname, ret_frames)

    return


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
