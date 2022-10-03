#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pathlib

import numpy as np

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
        # - simple driver
        wdirs = []
        for i in range(nframes):
            wdir = frames[i].info.get("wdir", f"cand{i}")
            wdirs.append(wdir)
        assert len(wdirs) == len(set(wdirs)), "Have duplicated structure names (wdir)..."

        # - run dynamics
        new_frames = []
        with CustomTimer(name="run-driver"):
            for wdir, atoms in zip(wdirs, frames):
                #print(wdirs, atoms)
                worker.driver.reset()
                worker.driver.directory = directory / wdir
                new_atoms = worker.driver.run(atoms)
                new_frames.append(new_atoms)
    else:
        # - interacts with scheduler
        worker.run(frames)
        worker.inspect()
        #if len(worker._get_unretrieved_jobs()) > 0:
        new_frames = worker.retrieve()

    # - report
    if new_frames:
        energies = [a.get_potential_energy() for a in new_frames]
        content = f"nframes: {len(new_frames)}\n"
        content += "statistics of total energies: min {:>12.4f} max {:>12.4f} avg {:>12.4f}".format(
            np.min(energies), np.max(energies), np.average(energies)
        )
        print(content)
        write(worker.directory/"new_frames.xyz", new_frames, append=True)

    return

if __name__ == "__main__":
    pass