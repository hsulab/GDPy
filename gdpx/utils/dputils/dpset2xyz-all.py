#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import shutil
import pathlib
import argparse
from ase.io.formats import F

import numpy as np

from ase.io import read, write

from gdpx.utils.data.dpsets import find_systems_set

from sklearn.model_selection import train_test_split, KFold


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-f', '--file',
        default='./test.xyz', help='xyz file'
    )
    parser.add_argument(
        '-nj', '--njobs', type=int,
        default=4, help='upper limit on number of directories'
    )
    parser.add_argument(
        '-ns', '--NotSave', action="store_false",
        help='save xyz for dpsets'
    )
    parser.add_argument(
        '-na', '--nall', action="store_false",
        help='not save train and test togather'
    )

    args = parser.parse_args()

    systems = []
    for p in pathlib.Path(args.file).glob("O*"):
        systems.append(p)
    print(systems)
    systems.sort()

    eann_dataset = pathlib.Path("./eann-dataset")

    for i, cur_system in enumerate(systems):
        cur_name = pathlib.Path(cur_system.stem) 
        print("system %d:" %i, cur_system)
        if not cur_name.exists():
            cur_name.mkdir()
        train_xyz = cur_name / (cur_system.stem + '-train.xyz')
        test_xyz = cur_name / (cur_system.stem + '-test.xyz')
        all_xyz = cur_name / (cur_system.stem + '-all.xyz')
        if train_xyz.exists() and test_xyz.exists():
            print('read existed frames...')
            train_frames = read(train_xyz, ':')
            test_frames = read(test_xyz, ':')
        else:
            print('read and save frames...')
            train_frames, test_frames = find_systems_set(cur_system)
            write(train_xyz, train_frames)
            write(test_xyz, test_frames)
        all_frames = []
        all_frames.extend(train_frames)
        all_frames.extend(test_frames)
        if not all_xyz.exists():
            write(all_xyz, all_frames)
        # merge data into eann-dataset
        eann_name = eann_dataset / cur_system.stem
        if not eann_name.exists():
            eann_name.mkdir()
        eann_file = eann_name / "train.xyz"
        eann_file_test = eann_name / "test.xyz"
        reduced_xyz = cur_name / (cur_system.stem + "reduced.xyz")
        #if reduced_xyz.exists():
        #    print("used reduced")
        #    shutil.copy(reduced_xyz, eann_file)
        #else:
        #    print("used original")
        #    shutil.copy(all_xyz, eann_file)
        nframes = len(all_frames)
        print("#frames: ", nframes)
        #kf = KFold(n_splits=5, shuffle=True, random_state=42)
        ##print(kf.get_n_splits(np.arange(nframes)))
        #for train_index, test_index in kf.split(np.arange(nframes)):
        #    print(len(train_index))
        #    print(train_index)
        #    print(len(test_index))
        #    print(test_index)
        train_index, test_index = train_test_split(np.arange(nframes), test_size=0.10, random_state=42)
        #print(len(train_index))
        #print(train_index)
        #print(len(test_index))
        #print(test_index)

        # train 
        new_train_frames = []
        for train_i in train_index:
            atoms = all_frames[train_i]
            atoms.info["index"] = train_i
            new_train_frames.append(atoms)
        write(eann_file, new_train_frames)
        print(len(new_train_frames))

        # test
        new_test_frames = []
        for test_i in test_index:
            atoms = all_frames[test_i]
            atoms.info["index"] = test_i
            new_test_frames.append(atoms)
        write(eann_file_test, new_test_frames)
        print(len(new_test_frames))

