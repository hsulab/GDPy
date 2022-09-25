#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pathlib

from ase.io import read, write

from GDPy.computation.worker.worker import AbstractWorker
from GDPy.potential.register import create_potter

from GDPy.utils.command import CustomTimer


DEFAULT_MAIN_DIRNAME = "MyWorker"


def run_worker(structure: str, directory=pathlib.Path.cwd()/DEFAULT_MAIN_DIRNAME, worker=None):
    """"""
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

    if worker.scheduler.name == "local":
        wdirs = []
        for i in range(nframes):
            wdir = frames[i].info.get("wdir", f"cand{i}")
            wdirs.append(wdir)

        # - run dynamics
        new_frames = []
        with CustomTimer(name="run-driver"):
            for wdir, atoms in zip(wdirs, frames):
                #print(wdirs, atoms)
                worker.driver.reset()
                worker.driver.directory = directory / wdir
                new_atoms = worker.driver.run(atoms)
                new_frames.append(new_atoms)
    
        # - report
        energies = [a.get_potential_energy() for a in new_frames]
        print(energies)
    else:
        worker.run(frames)
        worker.inspect()
        #if len(worker._get_unretrieved_jobs()) > 0:
        new_frames = worker.retrieve()
        if new_frames:
            write(worker.directory/"new_frames.xyz", new_frames, append=True)

    return

if __name__ == "__main__":
    pass