#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import itertools
import numbers
import pathlib
from typing import List

import h5py
import numpy as np

from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator

#:
GROUP_NAME = "array2d"

#:
RETAINED_INFO_NAMES: List[str] = [
    "confid", "step",
    "max_devi_e", "min_devi_e", "avg_devi_e",
    "max_devi_v", "min_devi_v", "avg_devi_v",
    "max_devi_f", "min_devi_f", "avg_devi_f"
]

#:
RETAIEND_INFO_DTYPES: List[str] = [
    "i8", "i8",
    "f", "f", "f",
    "f", "f", "f",
    "f", "f", "f"
]

#:
RETAINED_CALC_PROPS: List[str] = ["energy", "forces"]

def reconstruct_images_from_hd5grp(grp):
    """Reconstruct an atoms_array from data stored in HDF5 group `images`."""
    #print("keys: ", list(grp.keys()))
    # - rebuild structures
    images = []
    for box, pbc, atomic_numbers, positions in zip(grp["box"], grp["pbc"], grp["atype"], grp["positions"]):
        atoms = Atoms(numbers=atomic_numbers, positions=positions, cell=box.reshape(3,3), pbc=pbc)
        images.append(atoms)
    nimages = len(images)
        
    # -- add info
    for name in RETAINED_INFO_NAMES:
        data = grp.get(name, default=None)
        if data is not None:
            for atoms, v in zip(images, data):
                atoms.info[name] = v

    # -- add calc
    results = [{} for _ in range(nimages)] # List[Mapping[str,data]]
    for name in RETAINED_CALC_PROPS:
        data = grp.get(name, default=None)
        if data is not None:
            for i, v in enumerate(data):
                results[i][name] = v
        
    for atoms, ret in zip(images, results):
        spc = SinglePointCalculator(atoms, **ret)
        atoms.calc = spc
    
    return AtomsArray(images, dict(grp.attrs))


class AtomsArray:

    #: Atoms data.
    _images: List[Atoms] = None

    #: Metadata about how this array is created.
    metadata = None

    #: 
    properties: dict = {}

    #: Indices of structures that are selected.
    _markers: List[int] = None

    # MD and MIN settings... TODO: type?
    # dump period

    def __init__(self, images: List[Atoms], *args, **kwargs):
        """"""
        self._images = images

        return
    
    @property
    def images(self):
        """"""

        return self._images
    
    def save(self, target):
        """"""
        target = pathlib.Path(target)

        # - save to hd5
        with h5py.File(target, "w") as fopen:
            # -
            grp = fopen.create_group("images")
            self._save_to_hd5grp(grp)

        return
    
    def _save_to_hd5grp(self, grp):
        """"""
        # TODO: variable numebr of atoms?
        # box, pbc, atomic_numbers, positions, forces, velocities
        boxes = np.array(
            [a.get_cell(complete=True) for a in self._images], dtype=np.float64
        ).reshape(-1,9)
        pbcs = np.array(
            [a.get_pbc() for a in self._images], dtype=np.int8
        )
        atomic_numbers = np.array(
            [a.get_atomic_numbers() for a in self._images], dtype=np.int8
        )
        positions = np.array([a.get_positions() for a in self._images], dtype=np.float64)

        # -- sys props
        box_dset = grp.create_dataset("box", data=boxes, dtype="f8")
        pbc_dset = grp.create_dataset("pbc", data=pbcs, dtype="i8")
        atype_dset = grp.create_dataset("atype", data=atomic_numbers, dtype="i8")
        pos_dset = grp.create_dataset("positions", data=positions, dtype="f8")

        # - add some info
        #   confid, step
        #   max_devi_e, min_devi_e, avg_devi_e
        #   max_devi_v, min_devi_v, avg_devi_v
        #   max_devi_f, min_devi_f, avg_devi_f
        for name, dtype in zip(RETAINED_INFO_NAMES, RETAIEND_INFO_DTYPES):
            data = [a.info.get(name, np.nan) for a in self._images]
            if not np.all(np.isnan(data)):
                _ = grp.create_dataset(name, data=data, dtype=dtype)

        # -- calc props
        # TODO: without calc properties?
        energies = np.array([a.get_potential_energy() for a in self._images], dtype=np.float64)
        forces = np.array([a.get_forces() for a in self._images], dtype=np.float64)

        ene_dset = grp.create_dataset("energy", data=energies, dtype="f8")
        frc_dset = grp.create_dataset("forces", data=forces, dtype="f8")

        return
    
    @staticmethod
    def from_file(target):
        """"""
        with h5py.File(target, "r") as fopen:
            atoms_array = reconstruct_images_from_hd5grp(fopen.require_group("images"))

        return atoms_array
    
    @staticmethod
    def _from_hd5grp(grp_inp):
        """"""
        grp = grp_inp.require_group("images")

        return reconstruct_images_from_hd5grp(grp)
    
    @staticmethod
    def from_list(frames: List[Atoms]) -> "AtomsArray":

        return AtomsArray(images=frames)
    
    @property
    def markers(self):
        """Return markers.

        If it is the first time, all structures are returned.
        
        """
        if self._markers is None:
            self._markers = np.arange(len(self))

        return self._markers
    
    @markers.setter
    def markers(self, new_markers: List[int]):
        """"""
        nstructures = len(self)
        for i in new_markers:
            if i < 0 or i >= nstructures:
                raise IndexError("Mask index is out of range.")
            
        self._markers = sorted(new_markers)

        return

    def get_marked_structures(self):
        """"""
        markers = self.markers

        structures = []
        for i in markers:
            structures.append(self[i])

        return structures
    
    @property
    def shape(self):
        """"""
        max_natoms = max([len(a) for a in self._images])

        return (len(self),max_natoms)
    
    def __len__(self):
        """"""
        return len(self._images)
    
    def __getitem__(self, i):
        """"""
        if isinstance(i, numbers.Integral):
            nimages = len(self)
            if i < -nimages or i >= nimages:
                raise IndexError(f"Index {i} out of range {nimages}.")

            return self._images[i]
        else:
            raise ValueError("Index should be an integer")
    
    def __repr__(self) -> str:
        """"""

        return f"AtomsArray(task={self.task}, shape={len(self)}, markers={self.markers})"


class AtomsArray2D:

    """The base class of a 2D array of atoms.
    """

    name = "array2d"

    #: Structure data.
    _rows: List[AtomsArray] = []

    def __init__(self, rows: List[AtomsArray]=[], *args, **kwargs) -> None:
        """"""
        self._rows = rows

        return
    
    @staticmethod
    def from_file(target):
        """"""
        rows = []
        with h5py.File(target, "r") as fopen:
            grp = fopen.require_group(GROUP_NAME)
            indices = sorted([int(i) for i in grp.keys()])
            for i in indices:
                atoms_array = AtomsArray._from_hd5grp(grp[str(i)])
                rows.append(atoms_array)

        return AtomsArray2D(rows=rows)
    
    @staticmethod
    def from_list2d(frames_list: List[List[Atoms]]) -> "AtomsArray2D":
        """"""

        return AtomsArray2D(rows=[AtomsArray.from_list(x) for x in frames_list])
    
    @property
    def rows(self):
        """"""

        return self._rows
    
    def save_file(self, target):
        """"""
        if len(self) == 0:
            raise RuntimeError(f"AtomsArray do not have any atoms.")

        target = pathlib.Path(target).resolve()
        if target.exists():
            raise RuntimeError(f"{str(target)} exists.")

        with h5py.File(target, mode="w") as fopen:
            grp = fopen.create_group(GROUP_NAME)
            for i in range(len(self)):
                curr_grp = grp.create_group(f"{str(i)}/images")
                self._rows[i]._save_to_hd5grp(curr_grp)
            ...

        return
    
    def get_markers(self):
        """"""
        # markers looks like [(0, (1,2)), (2, (0))] traj number, and structure number
        markers = []
        for i, t in enumerate(self._rows):
            if len(t.markers) > 0:
                markers.append([i, t.markers])

        return markers
    
    def get_unpacked_markers(self):
        """"""
        markers = []
        for i, t in enumerate(self._rows):
            if len(t.markers) > 0:
                for j in t.markers:
                    markers.append((i, j))
        
        return markers
    
    def set_markers(self, new_markers):
        """Set markers.

        If there is no new markers for the current array, the old ones are cleared.

        """
        # new_markers [(0, (1,2)), (2, (0))] traj number, and structure number
        for i, s in new_markers:
            self._rows[i].markers = s
        
        # - clear
        new_ = [m[0] for m in new_markers]
        for i in range(len(self)):
            if i not in new_:
                self._rows[i].markers = []

        return

    def get_number_of_markers(self):
        """"""
        markers = self.get_markers()

        return len(list(itertools.chain(*[m[1] for m in markers])))
    
    def get_marked_structures(self):
        """Sync with unpacked markers."""

        #return list(itertools.chain(*[t.get_marked_structures() for t in self._rows]))
        return [self[i][j] for i, j in self.get_unpacked_markers()]
    
    def extend(self, iterable):
        """"""
        for x in iterable:
            self._rows.append(x)

        return
    
    @property
    def shape(self):
        """"""
        shape = [len(self)]
        shape.extend(self._rows[0].shape)

        return tuple(shape)
    
    @property
    def nstructures(self):
        """"""
        stats = [len(t) for t in self._rows]

        return sum(stats)
    
    def __len__(self):

        return len(self._rows)

    def __getitem__(self, i):
        """"""
        if isinstance(i, numbers.Integral):
            nrows = len(self)
            if i < -nrows or i >= nrows:
                raise IndexError(f"Index {i} out of range {nrows}.")

            return self._rows[i]
        else:
            raise ValueError("Index should be an integer")
    
    def __repr__(self) -> str:
        """"""

        return f"AtomsArray2D(number: {len(self)}, shape: {self.shape}, markers: {self.get_number_of_markers()})"


if __name__ == "__main__":
    ...