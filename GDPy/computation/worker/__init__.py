#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pathlib

from ase.io import read, write

from GDPy.computation.worker.worker import AbstractWorker
from GDPy.potential.register import create_potter


DEFAULT_MAIN_DIRNAME = "MyWorker"

def create_worker(params: dict):
    """"""
    worker = AbstractWorker(params)

    return worker

def run_worker(params: str, structure: str, potter=None):
    """"""
    worker = create_potter(params)
    
    # TODO: 
    frames = read(structure, ":")

    # - find input frames
    worker.directory = pathlib.Path.cwd() / DEFAULT_MAIN_DIRNAME

    worker.run(frames)
    worker.inspect()
    if len(worker._get_unretrieved_jobs()) > 0:
        new_frames = worker.retrieve()
        if new_frames:
            write(worker.directory/"new_frames.xyz", new_frames, append=True)

    return

if __name__ == "__main__":
    pass