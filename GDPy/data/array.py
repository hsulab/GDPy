#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import itertools
import functools
import operator
import numbers
import pathlib
from typing import Any, List, Mapping

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

    #: Array-like indices.
    _markers = None

    #:
    _ind_map: Mapping[int, int] = None

    def __init__(self, data: list = None, markers = None) -> None:
        """Init from a List^n object."""
        # TODO: Check data should be a list
        if data is None:
            data = []

        self._shape, self._data, self._markers, self._ind_map = self._process_data(data)
        
        # TODO: Check IndexError?
        if markers is not None:
            self.markers = markers
        
        return
    
    @staticmethod
    def _process_data(data_nd):
        """"""
        sizes = [[len(data_nd)]]
        def _flat_inhomo_data(items: list):
            """"""
            if isinstance(items, list):
                if not isinstance(items[0], Atoms):
                    sizes.append([len(item) for item in items])
                    items = _flat_inhomo_data(list(itertools.chain(*items)))
                else:
                    return items
            else:
                ...

            return items

        data_1d = [a for a in _flat_inhomo_data(data_nd) if a is not None]
        shape = tuple([max(s) for s in sizes])
        #print(f"sizes: {sizes}")
        #print(f"shape: {shape}")

        def assign_markers(arr, seq):
            #print(arr)
            #print(seq)
            if isinstance(arr, list): # assume it is a list
                if arr[0] is None: #arr.ndim == 1:
                    inds = []
                    for i, a in enumerate(seq):
                        if isinstance(a, Atoms):
                            inds.append(i)
                    m = len(arr)
                    for i in range(m):
                        if i in inds:
                            arr[i] = True
                        else:
                            arr[i] = False
                else:
                    for subarr, subseq in itertools.zip_longest(arr, seq, fillvalue=()):
                        assign_markers(subarr, subseq)
            else:
                ...
        
        raw_markers = np.full(shape, None).tolist()
        _ = assign_markers(raw_markers, data_nd)
        markers_1d = np.argwhere(raw_markers)
        ind_map = {_map_idx(loc, shape): i for i, loc in enumerate(markers_1d)}

        return shape, data_1d, markers_1d, ind_map

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
        if isinstance(new_markers, list):
            ...
        elif isinstance(new_markers, np.ndarray):
            new_markers = new_markers.tolist()
        else:
            raise ValueError("Index must be a list or a ndarray.")

        self._markers = np.array(sorted(new_markers))

        return

    def get_marked_structures(self, markers=None):
        """Get structures according to markers.

        If custom markers is None, `self._markers` will be used instead.

        """
        if markers is None:
            curr_markers = self.markers
        else:
            curr_markers = markers

        structures = [self._data[self._ind_map[_map_idx(loc, self.shape)]] for loc in curr_markers]

        return structures

    def tolist(self):
        """"""
        data_1d = np.full(self.shape, None).flatten().tolist()
        for k, v in self._ind_map.items():
            data_1d[k] = self._data[v]

        return _reshape_data(data_1d, self.shape)
    
    @classmethod
    def from_file(cls, target):
        """"""
        with h5py.File(target, "r") as fopen:
            grp = fopen.require_group("images")
            shape = grp.attrs["shape"]
            images = cls._from_hd5grp(grp=grp)
            markers = np.array(grp["markers"][:])
            mapper = {k: v for k, v in zip(grp["map_k"], grp["map_v"])}

        data_1d = np.full(shape, None).flatten().tolist()
        for k, v in mapper.items():
            data_1d[k] = images[v]
        
        data = _reshape_data(data_1d, shape=shape)

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

            # - save mapper
            mapper_k, mapper_v = [], []
            for k, v in self._ind_map.items():
                mapper_k.append(k)
                mapper_v.append(v)
            grp.create_dataset("map_k", data=mapper_k, dtype="i8")
            grp.create_dataset("map_v", data=mapper_v, dtype="i8")

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
            raise IndexError("Index must be an integer, a slice or a tuple.")
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
        #print(f"tshape: {tshape}")

        # - convert indices
        products = list(itertools.product(*indices))
        global_indices = [_map_idx(x, self._shape) for x in products]

        # - get data
        #print(f"tshape: {tshape}")
        ret_data = [self._data[self._ind_map[x]] for x in global_indices]
        if tshape:
            ret = _reshape_data(ret_data, tshape)
        else: # tshape is empty, means this is a single atoms
            ret = ret_data[0]

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