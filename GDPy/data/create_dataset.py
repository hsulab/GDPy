#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import pathlib
from joblib import Parallel, delayed

import numpy as np

from ase.io import read, write

from sklearn.model_selection import train_test_split, KFold

def read_system(sdir: pathlib.Path):
    """"""
    cur_frames = []
    for p in sdir.glob("*.xyz"):
        cur_frames.extend(read(p, ":"))

    return cur_frames


pattern = "O[0-9]*"

dataset = pathlib.Path(
    "/users/40247882/scratch2/PtOx-dataset"
)

partition_dataset = pathlib.Path(
    "/users/40247882/scratch2/PtOx-dataset/eann12"
)

system_dirs = []
for p in dataset.glob(pattern):
    system_dirs.append(p)
system_dirs.sort()
print("number of systems: ", len(system_dirs))

for p in system_dirs:
    print(str(p))

exit()

start_time = time.time()
#all_frames = []
all_nframes = 0
if True:
    for sdir in system_dirs:
        print("System ", sdir.stem)
        par_sys_dir = partition_dataset / sdir.name
        par_sys_dir.mkdir()
        train_file = par_sys_dir / "train.xyz"
        with open(train_file, "w") as fopen:
            fopen.write("")
        test_file = par_sys_dir / "test.xyz"
        with open(test_file, "w") as fopen:
            fopen.write("")

        # run over xyz files
        for p in sdir.glob("*.xyz"):
            # read and summary
            print("  subsystem: ", p.name)
            sub_frames = read(p, ":")
            nframes = len(sub_frames)
            train_index, test_index = train_test_split(np.arange(nframes), test_size=0.10, random_state=42)
            print(
                "  ntrain: %d ntest: %d ntotal: %d" %(len(train_index), len(test_index), nframes)
            )

            # train 
            new_train_frames = []
            for train_i in train_index:
                atoms = sub_frames[train_i]
                atoms.info["index"] = p.name + str(train_i)
                new_train_frames.append(atoms)
            write(train_file, new_train_frames, append=True)

            # test
            new_test_frames = []
            for test_i in test_index:
                atoms = sub_frames[test_i]
                atoms.info["index"] = p.name + str(test_i)
                new_test_frames.append(atoms)
            write(test_file, new_test_frames, append=True)

            # count
            all_nframes += nframes
        #print("System %s has %d frames..." %(sdir.stem, len(cur_frames)))
        #all_frames.extend(cur_frames)
        #exit()
else:
    frames_list = Parallel(n_jobs=4)(delayed(read_system)(p) for p in system_dirs)
    for frames in frames_list:
        all_frames.extend(frames)
end_time = time.time()

print("Total number of frames: ", all_nframes)
print("Time cost: ", end_time-start_time)


if __name__ == "__main__":
    pass