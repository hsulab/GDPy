#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import itertools
import functools
import operator
import numbers
import pathlib
from typing import Any, Optional, List, Mapping

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

#: Saved calculated property names.
RETAINED_CALC_PROPS: List[str] = ["energy", "free_energy", "forces"]

#: Saved calculated atomic property names.
RETAINED_ATOMIC_CALC_PROPS: List[str] = ["forces"]


def _flat_data(items):
    """"""
    # if not isinstance(ret, Atoms):
    if isinstance(items, list) and not isinstance(items[0], Atoms):
        items = _flat_data(list(itertools.chain(*items)))

    return items


def _reshape_data(data, shape):
    """"""
    data = data  # NOTE: A PURE LIST
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
    _data: Optional[List[Atoms]] = None

    #: Array shape.
    _shape: Optional[List[int]] = None

    #: Array-like indices.
    _markers = None

    #:
    _ind_map: Optional[Mapping[int, int]] = None

    """Define an atoms array object.

    This definition gives correct interval selection.

    """

    def __init__(self, data: list = None, markers=None) -> None:
        """Init from a List^n object."""
        # TODO: Check data should be a list
        if data is None:
            data = []
        # TODO: Check data should be a list
        if isinstance(data, list):
            data = data
        elif isinstance(data, AtomsNDArray):
            data = data.tolist()
        else:
            raise ValueError(
                f"Data should be a list or a AtomsNDArray instead of {type(data)}.")

        if len(data) == 0:
            raise RuntimeError(f"Input data is empty as {data}.")

        self._shape, self._data, self._markers, self._ind_map = self._process_data(
            data)

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
            if isinstance(items, list) or isinstance(items, tuple):
                # NOTE: The input items must be a nested list only with Atoms or None elements
                #print(f"current size: {sizes}")
                #print(f"current size: {items[0]}")
                if not (isinstance(items[0], Atoms) or items[0] is None):
                    sizes.append([len(item) for item in items])
                    items = _flat_inhomo_data(list(itertools.chain(*items)))
                else:
                    return items
            else:
                ...

            return items

        # NOTE: properly deal with None?
        data_1d = [a for a in _flat_inhomo_data(data_nd) if a is not None]
        shape = tuple([max(s) for s in sizes])
        # print(f"sizes: {sizes}")
        # print(f"shape: {shape}")

        def assign_markers(arr, seq):
            # print(arr)
            # print(seq)
            if isinstance(arr, list):  # assume it is a list
                if arr[0] is None:  # arr.ndim == 1:
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
    def ndim(self) -> int:
        """"""

        return len(self.shape)

    @property
    def raw_markers(self):
        """"""
        raw_markers = np.full(self.shape, None)
        for m in self.markers:
            raw_markers[tuple(m)] = True

        return raw_markers

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

        structures = [self._data[self._ind_map[_map_idx(
            loc, self.shape)]] for loc in curr_markers]

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
        natoms_list = grp["natoms"]

        images = []
        for natoms, box, pbc, atomic_numbers, positions in zip(
            natoms_list, grp["box"], grp["pbc"], grp["atype"], grp["positions"]
        ):
            atoms = Atoms(
                numbers=atomic_numbers[:natoms], positions=positions[:natoms, :],
                cell=box.reshape(3, 3), pbc=pbc
            )
            images.append(atoms)
        nimages = len(images)

        # -- add info
        for name in RETAINED_INFO_NAMES:
            data = grp.get(name, default=None)
            if data is not None:
                for atoms, v in zip(images, data):
                    atoms.info[name] = v

        # -- add calc
        results = [{} for _ in range(nimages)]  # List[Mapping[str,data]]
        for name in RETAINED_CALC_PROPS:
            data = grp.get(name, default=None)
            if data is not None:
                for i, v in enumerate(data):
                    if name not in RETAINED_ATOMIC_CALC_PROPS:
                        results[i][name] = v
                    else:
                        results[i][name] = v[:natoms_list[i]]

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

        NOTE:
            support variable number of atoms...

        """
        # - save structures
        nimages = len(images)
        natoms_list = np.array([len(a) for a in images], dtype=np.int32)
        boxes = np.array(
            [a.get_cell(complete=True) for a in images], dtype=np.float64
        ).reshape(-1, 9)
        pbcs = np.array(
            [a.get_pbc() for a in images], dtype=np.int8
        )
        # atomic_numbers = np.array(
        #    [a.get_atomic_numbers() for a in images], dtype=np.int8
        # )
        # positions = np.array([a.get_positions() for a in images], dtype=np.float64)
        atomic_numbers = np.zeros((nimages, max(natoms_list)), dtype=np.int32)
        positions = np.zeros((nimages, max(natoms_list), 3), dtype=np.float64)
        for i, a in enumerate(images):
            atomic_numbers[i, :natoms_list[i]] = a.get_atomic_numbers()
            positions[i, :natoms_list[i], :] = a.get_positions()

        # -- sys props
        natoms_dset = grp.create_dataset(
            "natoms", data=natoms_list, dtype="i8")
        box_dset = grp.create_dataset("box", data=boxes, dtype="f8")
        pbc_dset = grp.create_dataset("pbc", data=pbcs, dtype="i8")
        atype_dset = grp.create_dataset(
            "atype", data=atomic_numbers, dtype="i8")
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
        # NOTE: energy and forces have apply_constraint True...
        energies = np.array([a.get_potential_energy()
                            for a in images], dtype=np.float64)
        free_energies = []
        for i, a in enumerate(images):
            try:
                free_energy = a.get_potential_energy(force_consistent=True)
            except: # Assume no free_energy property is available.
                free_energy = energies[i]
            free_energies.append(free_energy)

        # forces = np.array([a.get_forces() for a in images], dtype=np.float64)
        forces = np.zeros((nimages, max(natoms_list), 3), dtype=np.float64)
        for i, a in enumerate(images):
            forces[i, :natoms_list[i], :] = a.get_forces()

        ene_dset = grp.create_dataset("energy", data=energies, dtype="f8")
        fen_dset = grp.create_dataset("free_energy", data=free_energies, dtype="f8")
        frc_dset = grp.create_dataset("forces", data=forces, dtype="f8")

        return

    # @classmethod
    # def squeeze(cls, axis=0):
    #    """Squeeze TODO: treat markers and map properly."""

    #    return cls()

    # def take(self, indices, axis=None):
    #    """"""

    #    return

    def __getitem__(self, key):
        """"""
        if isinstance(key, numbers.Integral) or isinstance(key, slice):
            key = [key] + [slice(None) for _ in range(len(self._shape)-1)]
        elif not isinstance(key, tuple):
            raise IndexError("Index must be an integer, a slice or a tuple.")
        assert len(key) <= len(self._shape), "Out of dimension."
        # BUG: <=?
        # print(f"key: {key}")

        # - get indices for each dimension
        indices, tshape = [], []
        for dim, i in enumerate(key):
            size = self._shape[dim]
            if isinstance(i, numbers.Integral):
                if i < -size or i >= size:
                    raise IndexError(
                        f"index {i} is out of bounds for axis {dim} with size {size}.")
                    # IndexError: index 1 is out of bounds for axis 0 with size 1
                if i < 0:
                    i += size
                curr_indices = [i]
            elif isinstance(i, slice):
                curr_indices = range(size)[i]
                for c_i in curr_indices:
                    if c_i <= -size or c_i >= size:
                        raise IndexError(f"Index {c_i} out of range {size}.")
                tshape.append(len(curr_indices))
            else:
                raise IndexError(
                    f"Index must be an integer or a slice for dimension {dim}.")
            indices.append(curr_indices)
        # print(f"tshape: {tshape}")

        # - convert indices
        products = list(itertools.product(*indices))
        global_indices = [_map_idx(x, self._shape) for x in products]
        # print(global_indices)
        # print(self._data)
        # print(f"nframes: {len(self._data)}")
        # print(self._ind_map)

        # - get data
        ret_data = []
        for x in global_indices:
            if x in self._ind_map:
                ret_data.append(self._data[self._ind_map[x]])
            else:
                ret_data.append(None)
        if tshape:
            ret = _reshape_data(ret_data, tshape)
        else:  # tshape is empty, means this is a single atoms
            ret = ret_data[0]

        return ret  # TODO: should this also be an array?

    def __len__(self):
        """"""
        return len(self._data)

    def __repr__(self) -> str:
        """"""

        return f"atoms_array(nimages: {len(self)}, shape: {self.shape})"


if __name__ == "__main__":
    ...
