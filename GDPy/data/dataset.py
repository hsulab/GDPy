#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
import itertools
import pathlib
from typing import List
import warnings

from ase import Atoms
from ase.io import read, write

from GDPy.core.register import registers
from GDPy.core.variable import Variable


@registers.dataloader.register
class XyzDataloader:

    name = "xyz"

    _print = print
    _debug = print

    """A directory-based dataset.

    There are several subdirs in the main directory. Each dirname follows the format that 
    `description-formula-type`, for example, `water-H2O-molecule`, is a system with structures 
    that have one single water molecule.

    """

    def __init__(self, dataset_path="./", batchsize=32, train_ratio=0.9, *args, **kwargs) -> None:
        """"""
        self.directory = pathlib.Path(dataset_path) # datset

        self.batchsize = batchsize
        self.train_ratio = train_ratio

        return
    
    def load(self):
        """Load dataset.

        All directories that have xyz files in `self.directory`.

        TODO:
            * Other file formats.

        """
        data_dirs = []
        def traverse_dirs(wdir):
            """"""
            for p in wdir.iterdir():
                if p.is_dir():
                    xyzpaths = list(p.glob("*.xyz"))
                    if len(xyzpaths) > 0:
                        data_dirs.append(p)
                    else:
                        traverse_dirs(p)
                else:
                    ...

            return
        traverse_dirs(self.directory)
        data_dirs = sorted(data_dirs)

        #print(len(data_dirs))
        #for p in data_dirs:
        #    print(str(p))

        return data_dirs
    
    def get_dataset(self):
        """"""
        data_dirs = self.load()

        self._print(data_dirs)
        self._print("\n--- auto data reader ---\n")

        batchsizes = self.batchsize
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
            self._print(str(cur_system))
            self._print(f"System {cur_system.stem} Batchsize {curr_batchsize}\n")
            frames = [] # all frames in this subsystem
            subsystems = list(cur_system.glob("*.xyz"))
            subsystems.sort() # sort by alphabet
            for p in subsystems:
                # read and split dataset
                p_frames = read(p, ":")
                p_nframes = len(p_frames)
                frames.extend(p_frames)
                self._print(f"  subsystem: {p.name} number {p_nframes}\n")

        return
    
    def transfer(self, frames: List[Atoms]):
        """Add structures into the dataset."""
        # - check chemical symbols
        system_dict = {} # {formula: [indices]}

        formulae = [a.get_chemical_formula() for a in frames]
        for k, v in itertools.groupby(enumerate(formulae), key=lambda x: x[1]):
            system_dict[k] = [x[0] for x in v]

        # - transfer data
        for formula, curr_indices in system_dict.items():
            # -- TODO: check system type
            system_type = self.system # currently, use user input one
            # -- name = description+formula+system_type
            dirname = "-".join([self.directory.parent.name, formula, system_type])
            target_subdir = self.target_dir/dirname
            target_subdir.mkdir(parents=True, exist_ok=True)

            # -- save frames
            curr_frames = [frames[i] for i in curr_indices]
            curr_nframes = len(curr_frames)

            strname = self.version + ".xyz"
            target_destination = self.target_dir/dirname/strname
            if not target_destination.exists():
                write(target_destination, curr_frames)
                self.pfunc(f"nframes {curr_nframes} -> {target_destination.name}")
            else:
                warnings.warn(f"{target_destination} exists.", UserWarning)

        return
    
    def as_dict(self):
        """"""
        dataset_params = {}
        dataset_params["name"] = self.name
        dataset_params["dataset_path"] = str(self.directory)
        dataset_params["batchsize"] = self.batchsize
        dataset_params["train_ratio"] = self.train_ratio

        dataset_params = copy.deepcopy(dataset_params)

        return dataset_params


if __name__ == "__main__":
    ...