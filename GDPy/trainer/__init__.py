#!/usr/bin/env python3
# -*- coding: utf-8 -*

import pathlib
from typing import NoReturn

from ase.io import read, write

from GDPy.computation.worker.train import TrainWorker


""""""

def run_trainer(pot_worker: TrainWorker, directory="./") -> NoReturn:
    """Train a potential using a TrainWorker

    Args:
        pot_worker: a potential TrainWorker
        directory: working dir for the training

    """
    # - prepare trainer
    train_dir = pathlib.Path(directory)
    pot_worker.directory = train_dir
    if not train_dir.exists():
        train_dir.mkdir()

    train_size = pot_worker.potter.train_size # number of models

    # - check dataset
    #   NOTE: assume it's a list of dirs that have xyz files
    frames = []
    dataset_dirs = pot_worker.potter.train_dataset
    for p in dataset_dirs:
        p = pathlib.Path(p)
        xyzfiles = list(p.glob("*.xyz"))
        for x in xyzfiles:
            frames.extend(read(x, ":"))
    print("nframes: ", len(frames))

    # - train
    #pot_worker._submit = False # TEST
    _ = pot_worker.run(frames, train_size)

    is_trained = True
    _ = pot_worker.retrieve()
    if pot_worker.get_number_of_running_jobs() > 0:
        is_trained = False
    else:
        # - update potter's calc
        print("UPDATE POTENTIAL...")
        pot_worker.potter.freeze(train_dir)

    return


if __name__ == "__main__":
    pass