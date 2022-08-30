#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pathlib

from ase.io import read, write

from GDPy.computation.worker.worker import AbstractWorker

from GDPy.utils.command import parse_input_file

DEFAULT_MAIN_DIRNAME = "MyWorker"

def create_worker(params: dict):
    """"""
    worker = AbstractWorker(params)

    return worker

def run_worker(params: str, structure: str, potter=None):
    """"""
    # - check if params are all valid
    params = parse_input_file(params)
    """ # TODO: check params
    if potter is None:
        pot_dict = params.get("potential", None)
        if pot_dict is None:
            raise RuntimeError("Need potential...")
        pm = PotManager() # main potential manager
        potter = pm.create_potential(pot_name = pot_dict["name"])
        potter.register_calculator(pot_dict["params"])
        potter.version = pot_dict["version"] # NOTE: important for calculation in exp
    """

    worker = AbstractWorker(params)

    # TODO: 
    frames = read(structure, ":")

    # - find input frames
    worker.directory = pathlib.Path.cwd() / DEFAULT_MAIN_DIRNAME

    worker.run(frames)
    if worker.get_number_of_running_jobs() > 0:
        new_frames = worker.retrieve()
        if new_frames:
            write(worker.directory/"new_frames.xyz", new_frames, append=True)

    return

if __name__ == "__main__":
    pass