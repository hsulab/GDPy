#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
import itertools
import pathlib
from typing import Union, List
import warnings

import numpy as np

from ase import Atoms
from ase.io import read, write
from ase.calculators.singlepoint import SinglePointCalculator

from ..core.register import registers
from ..core.variable import Variable
from ..core.node import AbstractNode

from .utils import is_a_valid_system_name


#: How to map keys in structures.
DEFAULT_PROP_MAP_KEYS: List[List[str]] = [("energy", "energy"), ("forces", "forces")]


def map_atoms_data(atoms: Atoms, prop_map_keys) -> None:
    """"""
    assert type(atoms.calc) == SinglePointCalculator, f"Atoms {atoms} has {atoms.calc}."

    # For some small dataset, structures can be used both as train and test,
    # thus, we need add a flag to avoid repeated mapping.
    is_mapped = atoms.info.get("is_mapped", False)

    if not is_mapped:
        results = copy.deepcopy(atoms.calc.results)
        for mapping_pairs in prop_map_keys:
            dst_key, src_key = mapping_pairs
            if src_key in results:
                # NOTE: The dst_data is overwritten if it exists
                src_val = results.pop(src_key)
                results[dst_key] = src_val
            else:
                raise KeyError(f"Atoms {atoms} has no key {src_key}.")
        atoms.calc.results = results
        atoms.info["is_mapped"] = True
    else:
        ...

    return


def traverse_xyzdirs(wdir):
    """"""
    data_dirs = []

    def recursive_traverse(wdir):
        for p in wdir.iterdir():
            if p.is_dir():
                xyzpaths = list(p.glob("*.xyz"))
                if len(xyzpaths) > 0:
                    data_dirs.append(p)
                recursive_traverse(p)
            else:
                ...
        return

    recursive_traverse(wdir)

    return data_dirs


class AbstractDataloader(AbstractNode): ...


@registers.dataloader.register
class XyzDataloader(AbstractDataloader):

    name = "xyz"

    """A directory-based dataset.

    There are several subdirs in the main directory. Each dirname follows the format that 
    `description-formula-type`, for example, `water-H2O-molecule`, is a system with structures 
    that have one single water molecule.

    """

    def __init__(
        self,
        dataset_path: Union[str, pathlib.Path] = "./",
        batchsize: int = 32,
        train_ratio: float = 0.9,
        random_seed: int = None,
        prop_keys: List[List[str]] = DEFAULT_PROP_MAP_KEYS,
        *args,
        **kwargs,
    ) -> None:
        """"""
        super().__init__(directory=dataset_path, random_seed=random_seed)

        self.batchsize = batchsize
        self.train_ratio = train_ratio

        self.prop_keys = prop_keys

        return

    def load(self) -> List[pathlib.Path]:
        """Load dataset.

        All directories that have xyz files in `self.directory`.

        TODO:
            * Other file formats.

        """
        data_dirs = traverse_xyzdirs(self.directory)
        data_dirs = sorted(data_dirs)

        return data_dirs

    def load_frames(self, *args, **kwargs):
        """"""
        data_dirs = traverse_xyzdirs(self.directory)
        data_dirs = sorted(data_dirs)

        names = [
            "+".join(str(x.relative_to(self.directory)).split("/")) for x in data_dirs
        ]

        nframes_tot, frames_list = 0, []
        for i, p in enumerate(data_dirs):
            curr_frames = []
            xyzpaths = sorted(list(p.glob("*.xyz")))
            for x in xyzpaths:
                curr_frames.extend(read(x, ":"))
            curr_nframes = len(curr_frames)
            nframes_tot += curr_nframes
            self._debug(f"{i:>4d} {str(p)} -> {len(curr_frames)}")
            frames_list.append(curr_frames)
        self._debug(f"Number of frames: {nframes_tot}")

        pairs = []
        for n, x in zip(names, frames_list):
            pairs.append([n, x])

        # - map keys
        should_map_keys = False
        for mapping_pairs in self.prop_keys:
            dst_key, src_key = mapping_pairs
            if dst_key != src_key:
                should_map_keys = True
            else:
                ...
        else:
            ...
        if should_map_keys:
            for n, x in pairs:
                for a in x:
                    map_atoms_data(a, self.prop_keys)

        return pairs

    def split_train_and_test(
        self,
    ):
        """Read structures and split them into train and test."""
        self._print("--- auto data reader ---")
        data_dirs = self.load()
        self._debug(data_dirs)

        # - aggregate data folders
        # dir_indices = list(range(data_dirs))
        system_paths = []
        for d in data_dirs:
            if is_a_valid_system_name(d.name):
                system_paths.append(d)
            else:
                system_paths.append(d.parent)

        system_groups = {}
        for k, v in itertools.groupby(enumerate(system_paths), key=lambda x: str(x[1])):
            if k not in system_groups:
                system_groups[k] = [data_dirs[e[0]] for e in v]
            else:
                system_groups[k].extend([data_dirs[e[0]] for e in v])

        # -- convert to tuple
        system_groups_ = []
        for k, v in system_groups.items():
            system_groups_.append([k, v])
        system_groups = system_groups_

        # -
        batchsizes = self.batchsize
        nsystems = len(system_groups)
        if isinstance(batchsizes, int):
            batchsizes = [batchsizes] * nsystems
        assert (
            len(batchsizes) == nsystems
        ), "Number of systems and batchsizes are inconsistent."

        # read configurations
        set_names = []
        train_size, test_size = [], []
        train_frames, test_frames = [], []
        adjusted_batchsizes = []  # auto-adjust batchsize based on nframes
        for i, (curr_system_group, curr_batchsize) in enumerate(
            zip(system_groups, batchsizes)
        ):
            curr_system = pathlib.Path(curr_system_group[0])
            set_name = "+".join(str(curr_system.relative_to(self.directory)).split("/"))
            set_names.append(set_name)
            self._print(f"System {set_name} Batchsize {curr_batchsize}")
            frames = []  # all frames in this subsystem
            for curr_subsystem in curr_system_group[1]:
                self._print(f"  {curr_subsystem.relative_to(curr_system)}")
                xyz_fpaths = list(curr_subsystem.glob("*.xyz"))
                xyz_fpaths.sort()  # sort by alphabet
                for p in xyz_fpaths:
                    # read and split dataset
                    p_frames = read(p, ":")
                    p_nframes = len(p_frames)
                    frames.extend(p_frames)
                    self._print(f"    subsystem: {p.name} number {p_nframes}")

            # split dataset and get adjusted batchsize
            # TODO: adjust batchsize of train and test separately
            nframes = len(frames)
            if nframes <= curr_batchsize:
                if nframes == 1 or curr_batchsize == 1:
                    new_batchsize = 1
                else:
                    new_batchsize = int(2 ** np.floor(np.log2(nframes)))
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
                    ntrain = int(
                        np.floor(nframes * self.train_ratio / new_batchsize)
                        * new_batchsize
                    )
                    train_index = self.rng.choice(nframes, ntrain, replace=False)
                    test_index = [x for x in range(nframes) if x not in train_index]
                adjusted_batchsizes.append(new_batchsize)

            ntrain, ntest = len(train_index), len(test_index)
            train_size.append(ntrain)
            test_size.append(ntest)

            self._print(
                f"    ntrain: {ntrain} ntest: {ntest} ntotal: {nframes} batchsize: {new_batchsize}"
            )

            curr_train_frames = [frames[train_i] for train_i in train_index]
            curr_test_frames = [frames[test_i] for test_i in test_index]

            # train
            train_frames.append(curr_train_frames)
            n_train_frames = sum([len(x) for x in train_frames])

            # test
            test_frames.append(curr_test_frames)
            n_test_frames = sum([len(x) for x in test_frames])
            self._print(
                f"  Current Dataset -> ntrain: {n_train_frames} ntest: {n_test_frames}"
            )

        assert len(train_size) == len(
            test_size
        ), "inconsistent train_size and test_size"
        train_size = sum(train_size)
        test_size = sum(test_size)
        self._print(f"Total Dataset -> ntrain: {train_size} ntest: {test_size}")

        # - map keys
        should_map_keys = False
        for mapping_pairs in self.prop_keys:
            dst_key, src_key = mapping_pairs
            if dst_key != src_key:
                should_map_keys = True
            else:
                ...
        else:
            ...

        if should_map_keys:
            for curr_frames in train_frames:
                for a in curr_frames:
                    map_atoms_data(a, self.prop_keys)
            for curr_frames in test_frames:
                for a in curr_frames:
                    map_atoms_data(a, self.prop_keys)

        return set_names, train_frames, test_frames, adjusted_batchsizes

    def transfer(self, frames: List[Atoms]):
        """Add structures into the dataset."""
        # - check chemical symbols
        system_dict = {}  # {formula: [indices]}

        formulae = [a.get_chemical_formula() for a in frames]
        for k, v in itertools.groupby(enumerate(formulae), key=lambda x: x[1]):
            system_dict[k] = [x[0] for x in v]

        # - transfer data
        for formula, curr_indices in system_dict.items():
            # -- TODO: check system type
            system_type = self.system  # currently, use user input one
            # -- name = description+formula+system_type
            dirname = "-".join([self.directory.parent.name, formula, system_type])
            target_subdir = self.target_dir / dirname
            target_subdir.mkdir(parents=True, exist_ok=True)

            # -- save frames
            curr_frames = [frames[i] for i in curr_indices]
            curr_nframes = len(curr_frames)

            strname = self.version + ".xyz"
            target_destination = self.target_dir / dirname / strname
            if not target_destination.exists():
                write(target_destination, curr_frames)
                self._print(f"nframes {curr_nframes} -> {target_destination.name}")
            else:
                warnings.warn(f"{target_destination} exists.", UserWarning)

        return

    def as_dict(self):
        """"""
        dataset_params = {}
        dataset_params["name"] = self.name
        dataset_params["dataset_path"] = str(self.directory.resolve())
        dataset_params["batchsize"] = self.batchsize
        dataset_params["train_ratio"] = self.train_ratio

        dataset_params = copy.deepcopy(dataset_params)

        return dataset_params


if __name__ == "__main__":
    ...
