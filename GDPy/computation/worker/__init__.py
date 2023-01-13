#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pathlib

import numpy as np

from ase.io import read, write

from GDPy.computation.worker.worker import AbstractWorker
from GDPy.potential.register import create_potter

from GDPy.utils.command import CustomTimer


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
        driver.run(atoms, read_exists=True, extra_info=None)
    
    ret_frames = []
    for wdir, atoms in zip(wdirs, frames):
        driver.directory = directory/wdir
        atoms = driver.read_converged()
        ret_frames.append(atoms)
    
    if o_fname is not None:
        write(directory/o_fname, ret_frames)

    return


def run_worker(structure: str, directory=pathlib.Path.cwd()/DEFAULT_MAIN_DIRNAME, local_exec=False, worker=None, o_fname=None):
    """"""
    directory = pathlib.Path(directory)

    # - read structures
    from GDPy.builder import create_generator
    generator = create_generator(structure)
    frames = generator.run()
    nframes = len(frames)
    print("nframes: ", nframes)

    #wdirs = params.pop("wdirs", None)
    #if wdirs is None:
    #    wdirs = [f"cand{i}" for i in range(nframes)]

    # - find input frames
    worker.directory = directory
    print(directory)

    worker.run(generator)
    #worker.inspect()
    #if len(worker._get_unretrieved_jobs()) > 0:

    # - report
    #new_frames = worker.retrieve()
    #if new_frames:
    #    energies = [a.get_potential_energy() for a in new_frames]
    #    content = f"nframes: {len(new_frames)}\n"
    #    content += "statistics of total energies: min {:>12.4f} max {:>12.4f} avg {:>12.4f}".format(
    #        np.min(energies), np.max(energies), np.average(energies)
    #    )
    #    print(content)
    #    #write(worker.directory/"new_frames.xyz", new_frames, append=True)
    if o_fname is not None:
        worker.inspect(resubmit=True)
        if worker.get_number_of_running_jobs() == 0:
            ret_frames = worker.retrieve()
            write(directory/o_fname, ret_frames)

    return

if __name__ == "__main__":
    pass