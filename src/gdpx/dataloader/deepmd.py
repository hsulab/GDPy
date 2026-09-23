#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import dataclasses
import pathlib
from typing import Optional, Union

import numpy as np
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import write


@dataclasses.dataclass
class DeepmdSystem:

    #: System name.
    name: str

    #: Frames.
    frames: list[Atoms]

    #: Number of train_and_split.
    train_and_split: tuple[int, int]

    #: Batchsize
    batchsize: int = 1

    def __post_init__(
        self,
    ):
        """"""
        self.nframes = len(self.frames)
        assert self.nframes == sum(self.train_and_split), f"{self.name}: {self.nframes} != sum({self.train_and_split})"

        composition_list = [a.get_chemical_formula() for a in self.frames]
        assert len(set(composition_list)) == 1, f"{self.name}: {composition_list[0]}?"
        self.composition = composition_list[0]

        return

    def write(self, directory: pathlib.Path) -> None:
        """"""
        directory.mkdir(parents=True, exist_ok=True)
        write(directory / f"{self.composition}.xyz", self.frames)

        return


class DeepmdDataloader:

    #: Datasset name.
    name: str = "deepmd"

    #:
    _systems: list[DeepmdSystem] = []

    def __init__(
        self,
        batchsize,
        batchsizes,
        cum_batchsizes,
        train_sys_dirs,
        valid_sys_dirs,
    ) -> None:
        """"""
        self.batchsize = batchsize
        self.batchsizes = batchsizes  # batchsize per system
        self.cum_batchsizes = cum_batchsizes
        self.train_sys_dirs = [str(x) for x in train_sys_dirs]
        self.valid_sys_dirs = [str(x) for x in valid_sys_dirs]

        return

    @staticmethod
    def from_directory(
        train_directory: Union[str, pathlib.Path],
        valid_directory: Optional[Union[str, pathlib.Path]] = None,
    ) -> "DeepmdDataloader":
        """"""
        # - read trainset...
        train_wdir = pathlib.Path(train_directory)
        trainset_dirs = sorted(train_wdir.iterdir())

        if valid_directory is not None:
            valid_wdir = pathlib.Path(valid_directory)
            validset_dirs = sorted(valid_wdir.iterdir())
        else:
            validset_dirs = []

        batchsize = 1
        batchsizes = [1] * len(trainset_dirs)
        cum_batchsizes = sum(batchsizes)

        return DeepmdDataloader(batchsize, batchsizes, cum_batchsizes, trainset_dirs, validset_dirs)

    @property
    def systems(
        self,
    ):
        """"""

        return self._systems

    def load(
        self,
    ) -> None:
        """"""
        systems = []
        for p in self.train_sys_dirs:
            p = pathlib.Path(p)
            print(p)
            curr_train_frames = DeepmdDataloader.convert_system_to_frames(p)
            num_curr_train_frames = len(curr_train_frames)
            curr_frames = curr_train_frames
            for p2 in self.valid_sys_dirs:
                p2 = pathlib.Path(p2)
                if p2.name == p.name:
                    curr_valid_frames = DeepmdDataloader.convert_system_to_frames(p2)
                    num_curr_valid_frames = len(curr_valid_frames)
                    curr_frames.extend(curr_valid_frames)
                    break
            else:
                num_curr_valid_frames = 0
            curr_system = DeepmdSystem(
                p.name,
                curr_frames,
                (num_curr_train_frames, num_curr_valid_frames),
            )
            systems.append(curr_system)
        self._systems = systems

        return

    def dump(self, directory: Union[str, pathlib.Path]) -> None:
        """"""
        directory = pathlib.Path(directory)
        for curr_system in self.systems:
            curr_system.write(
                directory / (curr_system.name + "-" + curr_system.composition),
            )
            print(
                f"{curr_system.name =} {curr_system.composition} {curr_system.train_and_split} {curr_system.nframes}"
            )

        return

    @staticmethod
    def set2frames(set_dir: pathlib.Path, chemical_symbols: list[str], pbc: list[int]) -> list[Atoms]:
        """Convert set into frames."""
        boxes = np.load(set_dir / "box.npy")
        nframes = boxes.shape[0]
        coords = np.load(set_dir / "coord.npy")
        energies = np.load(set_dir / "energy.npy")
        forces = np.load(set_dir / "force.npy")

        if nframes == 1:
            energies = energies.reshape(-1)

        frames = []
        for i in range(nframes):
            cell = boxes[i, :].reshape(3, 3)
            positions = coords[i, :].reshape(-1, 3)
            atoms = Atoms(
                symbols=chemical_symbols,
                positions=positions,
                cell=cell,
                pbc=pbc,
            )
            results = {
                "energy": energies[i],
                "forces": forces[i, :].reshape(-1, 3),
            }
            spc = SinglePointCalculator(atoms, **results)
            atoms.calc = spc
            frames.append(atoms)

        return frames

    @staticmethod
    def convert_system_to_frames(curr_system: pathlib.Path):
        """"""
        # find all set dirs
        set_dirs = sorted(curr_system.glob("set*"))

        type_list = np.loadtxt(curr_system / "type_map.raw", dtype=str)
        if len(type_list.shape) == 0:
            type_list = type_list.reshape(-1)
        atype = np.loadtxt(curr_system / "type.raw", dtype=int)
        if len(atype.shape) == 0:
            atype = atype.reshape(-1)
        chemical_symbols = [type_list[a] for a in atype]
        pbc = [1, 1, 1] if not (curr_system / "nopbc").exists() else [0, 0, 0]

        # train data
        frames = []
        for set_dir in set_dirs[:]:
            frames.extend(DeepmdDataloader.set2frames(set_dir, chemical_symbols, pbc))

        return frames

    def as_dict(
        self,
    ) -> dict:
        """"""
        params = {}
        params["name"] = self.name
        params["batchsize"] = self.batchsize
        params["batchsizes"] = self.batchsizes
        params["cum_batchsizes"] = self.cum_batchsizes
        params["train_sys_dirs"] = self.train_sys_dirs
        params["valid_sys_dirs"] = self.valid_sys_dirs

        return params


if __name__ == "__main__":
    ...
