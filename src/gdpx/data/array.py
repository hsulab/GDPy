#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import copy
import functools
import itertools
import numbers
import operator
import pathlib
from typing import Mapping, Optional, Union

import h5py
import numpy as np
import numpy.typing
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator

#: The retained keys in atoms.info.
RETAINED_INFO_NAMES: list[str] = [
    # fmt: off
    "confid", "step",
    "max_devi_e", "min_devi_e", "avg_devi_e",
    "max_devi_v", "min_devi_v", "avg_devi_v",
    "max_devi_f", "min_devi_f", "avg_devi_f"
]

#: The retained data types in atoms.info.
RETAIEND_INFO_DTYPES: list[str] = [
    # fmt: off
    "i8", "i8",
    "f", "f", "f",
    "f", "f", "f",
    "f", "f", "f"
]

#: Saved calculated property names.
RETAINED_CALC_PROPS: list[str] = ["energy", "free_energy", "forces"]

#: Saved calculated atomic property names.
RETAINED_ATOMIC_CALC_PROPS: list[str] = ["forces"]


def _reshape_data(data: list[Optional[Atoms]], shape: tuple[int, ...]) -> list:
    """"""
    for i, tsize in enumerate(shape[::-1][:-1]):
        npoints = len(data)
        length = int(npoints / tsize)
        data_ = []
        for i in range(length):
            data_.append(data[i * tsize : (i + 1) * tsize])
        data = data_

    return data


def _map_idx(loc: numpy.typing.NDArray, shape: tuple[int, ...]) -> int:
    """Map a location to an integer index based on the shape.

    Examples:
        >>> _map_idx([1, 3], (4, 8))
        11

    """
    i = 0
    for dim, j in enumerate(loc):
        i += j * functools.reduce(operator.mul, ([1] + list(shape[dim + 1 :])))

    return i


def _process_data(
    data_nd: list,
) -> tuple[
    tuple[int, ...], list[Atoms], numpy.typing.NDArray, Mapping[int, int]
]:
    """Process a nested list of Atoms."""
    sizes = [[len(data_nd)]]

    def _flat_inhomo_data(items: list) -> list:
        """Flatten the nested list.

        The input items must be a nested list only with Atoms or None elements.

        """
        if isinstance(items, list) or isinstance(items, tuple):
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

    def _assign_markers(arr, seq):
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
                for subarr, subseq in itertools.zip_longest(
                    arr, seq, fillvalue=()
                ):
                    _assign_markers(subarr, subseq)
        else:
            ...

    raw_markers = np.full(shape, None).tolist()
    _ = _assign_markers(raw_markers, data_nd)
    markers_1d = np.argwhere(raw_markers)
    ind_map = {_map_idx(loc, shape): i for i, loc in enumerate(markers_1d)}

    return shape, data_1d, markers_1d, ind_map


class AtomsNDArray:
    """Define an atoms array object.

    This definition gives correct interval selection.

    """

    def __init__(
        self,
        data: Optional[Union[list, "AtomsNDArray"]] = None,
        markers: Optional[numpy.typing.NDArray] = None,
    ) -> None:
        """Initialise an AtomsArray from a nested list.

        If the input data is an AtomsNDArray, the input markers will be overwritten by
        its current markers.

        """
        # Make the input data a list.
        if data is None:
            data = []
        if isinstance(data, list):
            data = data
        elif isinstance(data, AtomsNDArray):
            markers = data.markers  # overwrite input markers
            data = data.tolist()
        else:
            raise Exception(
                f"The input data should be a list or a AtomsNDArray instead of {type(data)}."
            )

        assert isinstance(
            data, list
        ), "The input data for AtomsNDArray should be a list."
        if len(data) == 0:
            raise Exception(f"The input data is empty as {data}.")

        self._shape, self._data, self._markers, self._ind_map = _process_data(
            data
        )

        self._init_markers = copy.deepcopy(self._markers)

        # Update markers with the custom input.
        if markers is not None:
            self.markers = markers

        return

    @property
    def shape(self) -> tuple[int, ...]:
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
    def markers(self) -> numpy.typing.NDArray:
        """Return markers.

        If it is the first time, all structures are returned.

        """

        return self._markers

    @markers.setter
    def markers(self, new_markers: Union[list, numpy.typing.NDArray]):
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

    @property
    def init_markers(self) -> numpy.typing.NDArray:
        """"""

        return self._init_markers

    def reset_markers(self) -> None:
        """"""
        self._markers = copy.deepcopy(self.init_markers)

        return

    def get_marked_structures(
        self, markers: Optional[numpy.typing.NDArray] = None
    ) -> list[Atoms]:
        """Get structures according to markers.

        If custom markers is None, `self._markers` will be used instead.

        """
        if markers is None:
            curr_markers = self.markers
        else:
            curr_markers = markers

        structures = [
            self._data[self._ind_map[_map_idx(loc, self.shape)]]
            for loc in curr_markers
        ]

        return structures

    def tolist(self) -> list:
        """"""
        data_1d = np.full(self.shape, None).flatten().tolist()
        for k, v in self._ind_map.items():
            data_1d[k] = self._data[v]

        return _reshape_data(data_1d, self.shape)

    @classmethod
    def from_file(cls, target: Union[str, pathlib.Path]):
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
                numbers=atomic_numbers[:natoms],
                positions=positions[:natoms, :],
                cell=box.reshape(3, 3),
                pbc=pbc,
            )
            images.append(atoms)
        nimages = len(images)

        # -- add info
        for name in RETAINED_INFO_NAMES:
            data = grp.get(name, default=None)
            if data is not None:
                for atoms, v in zip(images, data):
                    atoms.info[name] = v

        # add some extra properties (momenta, charges, ...)
        data = grp.get("momenta", default=None)
        if data is not None:
            for i, v in enumerate(data):
                a_v = v[: natoms_list[i]]
                if not np.all(np.isnan(a_v)):
                    images[i].set_momenta(a_v)

        # -- add calc
        results = [{} for _ in range(nimages)]  # list[Mapping[str,data]]
        for name in RETAINED_CALC_PROPS:
            data = grp.get(name, default=None)
            if data is not None:
                for i, v in enumerate(data):
                    if name not in RETAINED_ATOMIC_CALC_PROPS:
                        results[i][name] = v
                    else:
                        results[i][name] = v[: natoms_list[i]]

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

    def _convert_images(self, grp, images: list[Atoms]):
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
        pbcs = np.array([a.get_pbc() for a in images], dtype=np.int8)
        # atomic_numbers = np.array(
        #    [a.get_atomic_numbers() for a in images], dtype=np.int8
        # )
        # positions = np.array([a.get_positions() for a in images], dtype=np.float64)
        atomic_numbers = np.zeros((nimages, max(natoms_list)), dtype=np.int32)
        positions = np.zeros((nimages, max(natoms_list), 3), dtype=np.float64)
        for i, a in enumerate(images):
            atomic_numbers[i, : natoms_list[i]] = a.get_atomic_numbers()
            positions[i, : natoms_list[i], :] = a.get_positions()

        # -- sys props
        natoms_dset = grp.create_dataset(
            "natoms", data=natoms_list, dtype="i8"
        )
        box_dset = grp.create_dataset("box", data=boxes, dtype="f8")
        pbc_dset = grp.create_dataset("pbc", data=pbcs, dtype="i8")
        atype_dset = grp.create_dataset(
            "atype", data=atomic_numbers, dtype="i8"
        )
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
        energies = np.array(
            [a.get_potential_energy() for a in images], dtype=np.float64
        )
        free_energies = []
        for i, a in enumerate(images):
            try:
                free_energy = a.get_potential_energy(force_consistent=True)
            except:  # Assume no free_energy property is available.
                free_energy = energies[i]
            free_energies.append(free_energy)

        # forces = np.array([a.get_forces() for a in images], dtype=np.float64)
        forces = np.zeros((nimages, max(natoms_list), 3), dtype=np.float64)
        momenta = np.empty((nimages, max(natoms_list), 3), dtype=np.float64)
        momenta.fill(np.nan)
        for i, a in enumerate(images):
            forces[i, : natoms_list[i], :] = a.get_forces()
            if "momenta" in a.arrays:
                momenta[i, : natoms_list[i], :] = a.get_momenta()
            else:
                ...

        # save properties to dataset
        ene_dset = grp.create_dataset("energy", data=energies, dtype="f8")
        fen_dset = grp.create_dataset(
            "free_energy", data=free_energies, dtype="f8"
        )
        frc_dset = grp.create_dataset("forces", data=forces, dtype="f8")
        mom_dset = grp.create_dataset("momenta", data=momenta, dtype="f8")

        return

    # @classmethod
    # def squeeze(cls, axis=0):
    #    """Squeeze TODO: treat markers and map properly."""

    #    return cls()

    # def take(self, indices, axis=None):
    #    """"""

    #    return

    def __getitem__(self, key) -> Union[Atoms, list[Atoms]]:
        """"""
        if isinstance(key, numbers.Integral) or isinstance(key, slice):
            key = [key] + [slice(None) for _ in range(len(self._shape) - 1)]
        elif not isinstance(key, tuple):
            raise IndexError("Index must be an integer, a slice or a tuple.")
        if len(key) > len(self._shape):
            raise Exception(f"{key} is out of dimension of {self._shape}.")

        # - get indices for each dimension
        indices, tshape = [], []
        for dim, i in enumerate(key):
            size = self._shape[dim]
            if isinstance(i, numbers.Integral):
                i = int(i)
                if i < -size or i >= size:
                    raise IndexError(
                        f"Index {i} is out of bounds for axis {dim} with size {size}."
                    )
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
                    f"Index must be an integer or a slice for dimension {dim}."
                )
            indices.append(curr_indices)

        # Convert indices
        products = np.array(list(itertools.product(*indices)))
        global_indices = [_map_idx(x, self._shape) for x in products]

        # Get atoms
        ret_data = []
        for x in global_indices:
            if x in self._ind_map:
                ret_data.append(self._data[self._ind_map[x]])
            else:
                ret_data.append(None)
        if tshape:
            ret = _reshape_data(ret_data, tuple(tshape))
        else:  # tshape is empty, means this is a single atoms
            ret = ret_data[0]

        return ret

    def __len__(self) -> int:
        """"""
        return len(self._data)

    def __repr__(self) -> str:
        """"""

        return f"atoms_array(nimages: {len(self)}, shape: {self.shape})"


if __name__ == "__main__":
    ...
