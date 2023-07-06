#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import itertools
import functools
import operator
import numbers
import pathlib
from typing import Any, List

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

def _flat_data(items):
    """"""
    #if not isinstance(ret, Atoms):
    if isinstance(items, list) and not isinstance(items[0], Atoms):
        items = _flat_data(list(itertools.chain(*items)))

    return items


def _reshape_data(data, shape):
    """"""
    data = data # NOTE: A PURE LIST
    for i, tsize in enumerate(shape[::-1][:-1]):
        npoints = len(data)
        length = int(npoints/tsize)
        data_ = []
        for i in range(length):
            data_.append(data[i*tsize:(i+1)*tsize])
        data = data_

    return data


def _map_idx(loc, shape):
    """"""
    i = 0
    for dim, j in enumerate(loc):
        i += j*functools.reduce(operator.mul, ([1]+list(shape[dim+1:])))
    return i


class AtomsNDArray:

    #: Atoms data.
    _data: List[Atoms] = None

    #: Array shape.
    _shape: List[int] = None

    #: Has the same shape as the array.
    _markers = None

    def __init__(self, data: list = None, markers = None) -> None:
        """Init from a List^n object."""
        if data is None:
            data = []
        
        self._shape = self._get_shape(data)
        self._data = _flat_data(data)

        # TODO: Check IndexError?
        if markers is None:
            self._markers = np.argwhere(np.full(self._shape, True))
        else:
            self._markers = markers
        
        return
    
    @property
    def data(self):
        """"""

        return _reshape_data(self._data)
    
    def _get_shape(self, data):
        """"""
        dimensions = []
        def _get_depth(items: list) -> int:
            depth = 1 if isinstance(items, list) else 0
            if depth:
                curr_size = len(items)
                dimensions.append(curr_size)
                #for item in items:
                #    if isinstance(item, list):
                #        depth = max(depth, _get_depth(item)+1)
                types = [isinstance(item, list) for item in items]
                if all(types):
                    subsizes = [len(item) for item in items]
                    assert len(set(subsizes)) == 1, f"Inhomogeneous part found at {dimensions}+?."
                    depth = max(depth, _get_depth(items[0])+1)
                elif any(types):
                    raise RuntimeError("Found mixed List and others.")
                else:
                    ...
            else:
                return depth
            return depth
        depth = _get_depth(data)
        #print(f"depth: {depth}")
        #print(f"dimensions: {dimensions}")

        return tuple(dimensions)
    
    @property
    def shape(self):
        """"""

        return self._shape
    
    @property
    def markers(self):
        """Return markers.

        If it is the first time, all structures are returned.
        
        """

        return self._markers
    
    @markers.setter
    def markers(self, new_markers):
        """Set new markers.

        Args:
            new_markers: These should have the shape as the array.

        """
        # TODO: IndexError?
        self._markers = np.array(new_markers)

        return

    def get_marked_structures(self):
        """"""
        structures = [self._data[_map_idx(loc, self.shape)] for loc in self.markers]

        return structures
    
    @classmethod
    def from_file(cls, target):
        """"""
        with h5py.File(target, "r") as fopen:
            grp = fopen.require_group("images")
            shape = grp.attrs["shape"]
            images = cls._from_hd5grp(grp=grp)
            markers = np.array(grp["markers"][:])
        
        data = _reshape_data(images, shape=shape)

        return cls(data=data, markers=markers)
    
    @classmethod
    def _from_hd5grp(cls, grp):
        """Reconstruct an atoms_array from data stored in HDF5 group `images`."""
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
        
        return images
    
    def save_file(self, target):
        """"""
        with h5py.File(target, mode="w") as fopen:
            grp = fopen.create_group("images")
            grp.attrs["shape"] = self.shape

            # - save structures
            self._convert_images(grp=grp, images=self._data)

            # - save markers
            grp.create_dataset("markers", data=self.markers, dtype="i8")

        return
    
    def _convert_images(self, grp, images: List[Atoms]):
        """Convert data...

        TODO:
            support variable number of atoms...

        """
        # - save structures
        natoms_list = np.array([len(a) for a in images], dtype=np.int32)
        boxes = np.array(
            [a.get_cell(complete=True) for a in images], dtype=np.float64
        ).reshape(-1,9)
        pbcs = np.array(
            [a.get_pbc() for a in images], dtype=np.int8
        )
        atomic_numbers = np.array(
            [a.get_atomic_numbers() for a in images], dtype=np.int8
        )
        positions = np.array([a.get_positions() for a in images], dtype=np.float64)

        # -- sys props
        natoms_dset = grp.create_dataset("natoms", data=natoms_list, dtype="i8")
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
            data = [a.info.get(name, np.nan) for a in images]
            if not np.all(np.isnan(data)):
                _ = grp.create_dataset(name, data=data, dtype=dtype)

        # -- calc props
        # TODO: without calc properties?
        energies = np.array([a.get_potential_energy() for a in images], dtype=np.float64)
        forces = np.array([a.get_forces() for a in images], dtype=np.float64)

        ene_dset = grp.create_dataset("energy", data=energies, dtype="f8")
        frc_dset = grp.create_dataset("forces", data=forces, dtype="f8")

        return
    
    def __getitem__(self, key):
        """"""
        if isinstance(key, numbers.Integral) or isinstance(key, slice):
            key = [key] + [slice(None) for _ in range(len(self._shape)-1)]
        elif not isinstance(key, tuple):
            raise IndentationError("Index must be an integer, a slice or a tuple.")
        assert len(key) <= len(self._shape), "Out of dimension."
        #print(f"key: {key}")

        # - get indices for each dimension
        indices, tshape = [], []
        for dim, i in enumerate(key):
            size = self._shape[dim]
            if isinstance(i, numbers.Integral):
                if i <= -size or i >= size:
                    raise IndexError(f"Index {i} out of range {size}.")
                    # IndexError: index 1 is out of bounds for axis 0 with size 1
                curr_indices = [i]
            elif isinstance(i, slice):
                curr_indices = range(size)[i]
                for c_i in curr_indices:
                    if c_i <= -size or c_i >= size:
                        raise IndexError(f"Index {c_i} out of range {size}.")
                tshape.append(len(curr_indices))
            else:
                raise IndexError(f"Index must be an integer or a slice for dimension {dim}.")
            indices.append(curr_indices)

        # - convert indices
        products = list(itertools.product(*indices))
        global_indices = [_map_idx(x, self._shape) for x in products]

        # - get data
        #print(f"tshape: {tshape}")
        ret_data = [self._data[x] for x in global_indices]
        ret = _reshape_data(ret_data, tshape)

        return ret
    
    def __len__(self):
        """"""
        return len(self._data)
    
    def __repr__(self) -> str:
        """"""

        return f"atoms_array(nimages: {len(self)}, shape: {self.shape})"


class AtomsArray:

    #: Metadata about how this array is created.
    metadata = None

    #: 
    properties: dict = {}

    #: Indices of structures that are selected.
    _markers: List[int] = None

    # MD and MIN settings... TODO: type?
    # dump period

    #: Atoms data.
    _images: List[Atoms] = None

    def __init__(self, images: List[Atoms] = None, *args, **kwargs):
        """"""
        #: Atoms data.
        if images is None:
            self._images = []
        else:
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
        if len(self) > 0:
            return f"AtomsArray(shape={len(self)}, markers={self.markers})"
        else:
            return f"AtomsArray(shape={len(self)})"


class AtomsArray2D:

    """The base class of a 2D array of atoms.
    """

    name = "array2d"

    #: Structure data.
    _rows: List[AtomsArray] = None

    def __init__(self, rows: List[AtomsArray]=None, *args, **kwargs) -> None:
        """"""
        #: Structure data.
        if rows is None:
            self._rows = []
        else:
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

        if len(self) > 0:
            return f"AtomsArray2D(number: {len(self)}, shape: {self.shape}, markers: {self.get_number_of_markers()})"
        else:
            return f"AtomsArray2D(number: {len(self)})"


if __name__ == "__main__":
    ...