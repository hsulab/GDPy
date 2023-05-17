#!/usr/bin/env python3
# -*- coding: utf-8 -*

import itertools
import pathlib
from typing import NoReturn, List, Union

import numpy as np

from ase.io import read, write

from GDPy.computation.worker.train import TrainWorker


""""""
def split_dataset(
    data_dirs: List[Union[str,pathlib.Path]], batchsizes=16, train_ratio=0.9,
    reduce_system=False, rng=np.random
):
    """
    """
    print("\n--- auto data reader ---\n")
    #content += f"Use batchsize {batchsize} and train-ratio {train_ratio}\n"

    nsystems = len(data_dirs)
    if isinstance(batchsizes, int):
        batchsizes = [batchsizes]*nsystems
    assert len(batchsizes) == nsystems, "Number of systems and batchsizes are inconsistent."

    # read configurations
    set_names = []
    train_size, test_size = [], []
    train_frames, test_frames = [], []
    adjusted_batchsizes = [] # auto-adjust batchsize based on nframes
    for i, (cur_system, curr_batchsize) in enumerate(zip(data_dirs, batchsizes)):
        cur_system = pathlib.Path(cur_system)
        set_names.append(cur_system.name)
        print(f"System {cur_system.stem} Batchsize {curr_batchsize}\n")
        frames = [] # all frames in this subsystem
        subsystems = list(cur_system.glob("*.xyz"))
        subsystems.sort() # sort by alphabet
        for p in subsystems:
            # read and split dataset
            p_frames = read(p, ":")
            p_nframes = len(p_frames)
            frames.extend(p_frames)
            print(f"  subsystem: {p.name} number {p_nframes}\n")

        # split dataset and get adjusted batchsize
        # TODO: adjust batchsize of train and test separately
        nframes = len(frames)
        if nframes <= curr_batchsize:
            if nframes == 1 or curr_batchsize == 1:
                new_batchsize = 1
            else:
                new_batchsize = int(2**np.floor(np.log2(nframes)))
            adjusted_batchsizes.append(new_batchsize)
            # NOTE: use same train and test set
            #       since they are very important structures...
            train_index = list(range(nframes))
            test_index = list(range(nframes))
        else:
            if nframes == 1 or curr_batchsize == 1:
                new_batchsize = 1
                train_index = list(range(nframes))
                test_index = list(range(nframes))
            else:
                new_batchsize = curr_batchsize
                # - assure there is at least one batch for test
                #          and number of train frames is integer times of batchsize
                ntrain = int(np.floor(nframes * train_ratio / new_batchsize) * new_batchsize)
                train_index = rng.choice(nframes, ntrain, replace=False)
                test_index = [x for x in range(nframes) if x not in train_index]
            adjusted_batchsizes.append(new_batchsize)

        ntrain, ntest = len(train_index), len(test_index)
        train_size.append(ntrain)
        test_size.append(ntest)

        print(f"    ntrain: {ntrain} ntest: {ntest} ntotal: {nframes} batchsize: {new_batchsize}\n")

        curr_train_frames = [frames[train_i] for train_i in train_index]
        curr_test_frames = [frames[test_i] for test_i in test_index]
        if reduce_system:
            # train
            train_frames.extend(curr_train_frames)
            n_train_frames = len(train_frames)

            # test
            test_frames.extend(curr_test_frames)
            n_test_frames = len(test_frames)
        else:
            # train
            train_frames.append(curr_train_frames)
            n_train_frames = sum([len(x) for x in train_frames])

            # test
            test_frames.append(curr_test_frames)
            n_test_frames = sum([len(x) for x in test_frames])
        print(f"  Current Dataset -> ntrain: {n_train_frames} ntest: {n_test_frames}\n\n")

    assert len(train_size) == len(test_size), "inconsistent train_size and test_size"
    train_size = sum(train_size)
    test_size = sum(test_size)
    print(f"Total Dataset -> ntrain: {train_size} ntest: {test_size}\n")

    return set_names, train_frames, test_frames, adjusted_batchsizes

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
    # TODO: get random splits for each model instance
    (set_names, train_frames, test_frames, new_batchsizes) = split_dataset(
        pot_worker.potter.train_dataset, 
        batchsizes=pot_worker.potter.train_batchsize, 
        train_ratio=pot_worker.potter.train_split_ratio,
        rng=np.random.default_rng(seed=pot_worker.potter.train_split_seed)
    )

    # - train
    #pot_worker._submit = False # TEST
    _ = pot_worker.run([set_names, train_frames, test_frames, new_batchsizes], train_size)

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