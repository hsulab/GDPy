import functools
import itertools
import numbers
import operator
import pathlib
import traceback
from typing import Mapping, Optional, Union

import h5py
import numpy as np
import numpy.typing
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator

#: The retained keys in atoms.info.
RETAINED_INFO_NAMES: list[str] = [
    "confid", "step",
    "max_devi_e", "min_devi_e", "avg_devi_e",
    "max_devi_v", "min_devi_v", "avg_devi_v",
    "max_devi_f", "min_devi_f", "avg_devi_f",
]

#: The retained data types in atoms.info.
RETAINED_INFO_DTYPES: list[str] = [
    "i8", "i8",
    "f", "f", "f",
    "f", "f", "f",
    "f", "f", "f",
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
) -> tuple[tuple[int, ...], list[Atoms], numpy.typing.NDArray, Mapping[int, int]]:
    """Process a nested list of Atoms.

    Returns:
        shape: bounding-box shape of the nested list
        data_1d: flat list of non-None Atoms objects
        markers: boolean ndarray of shape ``shape`` — True where data exists
        ind_map: dict mapping flat index in the bounding box to position in data_1d

    """
    sizes = [[len(data_nd)]]

    def _flat_inhomo_data(items: list) -> list:
        if isinstance(items, (list, tuple)):
            if not (isinstance(items[0], Atoms) or items[0] is None):
                sizes.append([len(item) for item in items])
                items = _flat_inhomo_data(list(itertools.chain(*items)))
            else:
                return items

        return items

    data_1d = [a for a in _flat_inhomo_data(data_nd) if a is not None]
    shape = tuple([max(s) for s in sizes])

    # Build boolean markers (True where data exists)
    markers = np.full(shape, False)

    def _assign_markers(arr, seq):
        if isinstance(arr, list):
            if arr[0] is None or isinstance(arr[0], (Atoms, type(None))):
                for i, a in enumerate(seq):
                    if isinstance(a, Atoms):
                        arr[i] = True
                    else:
                        arr[i] = False
            else:
                for subarr, subseq in itertools.zip_longest(arr, seq, fillvalue=()):
                    _assign_markers(subarr, subseq)
        else:
            ...

    raw_list = np.full(shape, None).tolist()
    _assign_markers(raw_list, data_nd)

    # Convert nested list of bools to flat boolean ndarray
    for idx in np.ndindex(shape):
        elem = raw_list
        for d in idx:
            elem = elem[d]
        if elem:
            markers[idx] = True

    # Build ind_map (flat index -> data index)
    ind_map = {f: i for i, f in enumerate(np.flatnonzero(markers))}

    return shape, data_1d, markers, ind_map


class AtomsNDArray:
    """Define an N-dimensional atoms array object with labeled axes.

    Stores ASE Atoms objects in an N-dimensional grid with sparse storage
    (only non-None objects are kept in memory). Supports labeled dimensions
    and coordinate-based selection via ``.sel()`` and ``.loc``.

    """

    _dims: tuple[str, ...]

    def __init__(
        self,
        data: Optional[Union[list, "AtomsNDArray"]] = None,
        markers: Optional[Union[numpy.typing.NDArray, list]] = None,
        *,
        dims: Optional[Union[tuple[Optional[str], ...], list[Optional[str]]]] = None,
        coords: Optional[Mapping[str, numpy.typing.ArrayLike]] = None,
    ) -> None:
        """Initialise an AtomsNDArray.

        Args:
            data: Nested list of Atoms (or None for empty cells), or another
                AtomsNDArray (deep-copied).
            markers: Initial selection markers. Can be a boolean array of shape
                matching the data, or an (M, ndim) integer coordinate array
                (old format, auto-converted).
            dims: Names for each axis. Defaults to ``("dim_0", "dim_1", ...)``.
            coords: Coordinate labels per named axis,
                e.g. ``{"temperature": [300, 400, 500]}``.

        """
        if data is None:
            data = []
        if isinstance(data, list):
            data = data
        elif isinstance(data, AtomsNDArray):
            markers = data.markers if markers is None else markers
            dims = dims if dims is not None else data.dims
            coords = coords if coords is not None else data.coords
            data = data.tolist()
        else:
            raise TypeError(
                f"The input data should be a list or an AtomsNDArray "
                f"instead of {type(data)}."
            )

        assert isinstance(data, list), "The input data for AtomsNDArray should be a list."
        if len(data) == 0:
            raise ValueError(f"The input data is empty as {data}.")

        self._shape, self._data, self._mask, self._ind_map = _process_data(data)

        # Initialise markers from structure mask
        self._markers = self._mask.copy()
        self._init_markers = self._mask.copy()

        # Initialise dimension names
        if dims is not None:
            dims = tuple(dims)
            if len(dims) != self.ndim:
                raise ValueError(
                    f"Number of dims ({len(dims)}) does not match ndim ({self.ndim})."
                )
            self._dims = tuple(str(d) for d in dims)
        else:
            self._dims = tuple(f"dim_{i}" for i in range(self.ndim))

        # Initialise coordinate labels
        self._coords: dict[str, np.ndarray] = {}
        if coords is not None:
            for name, labels in coords.items():
                if name not in self._dims:
                    raise KeyError(
                        f"Coordinate name '{name}' not found in dims {self._dims}."
                    )
                axis = self._dims.index(name)
                labels_arr = np.asarray(labels)
                if len(labels_arr) != self._shape[axis]:
                    raise ValueError(
                        f"Coordinate '{name}' has {len(labels_arr)} labels "
                        f"but axis {axis} has size {self._shape[axis]}."
                    )
                self._coords[name] = labels_arr

        # Apply custom markers
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
        return len(self._shape)

    @property
    def dims(self) -> tuple[str, ...]:
        """Return dimension names for each axis."""
        return self._dims

    @property
    def coords(self) -> Mapping[str, np.ndarray]:
        """Return coordinate labels per named dimension."""
        return dict(self._coords)

    @property
    def markers(self) -> numpy.typing.NDArray:
        """Return the boolean selection mask."""
        return self._markers

    @markers.setter
    def markers(self, new_markers: Union[list, numpy.typing.NDArray]):
        """Set selection markers.

        Accepts a boolean array of shape ``self.shape``, or an integer
        coordinate array of shape ``(M, ndim)`` (old format, auto-converted).

        """
        new_markers = np.asarray(new_markers)

        if new_markers.dtype == bool or new_markers.dtype == np.bool_:
            if new_markers.shape != self._shape:
                raise ValueError(
                    f"Boolean markers shape {new_markers.shape} does not "
                    f"match array shape {self._shape}."
                )
            if not np.all(np.logical_not(new_markers) | self._mask):
                raise ValueError(
                    "Markers can only be True for positions that have data."
                )
            self._markers = new_markers.copy()
        elif new_markers.ndim == 2 and new_markers.shape[1] == self.ndim:
            # Integer coordinate array (old format) — convert to boolean
            bool_markers = np.full(self._shape, False)
            for coord in new_markers:
                bool_markers[tuple(coord)] = True
            if not np.all(np.logical_not(bool_markers) | self._mask):
                raise ValueError(
                    "Markers reference positions that have no data."
                )
            self._markers = bool_markers
        else:
            raise ValueError(
                "Markers must be a boolean array matching self.shape, "
                "or an (M, ndim) integer array."
            )

        return

    @property
    def init_markers(self) -> numpy.typing.NDArray:
        """Return the initial selection mask at construction time."""
        return self._init_markers

    def reset_markers(self) -> None:
        """Reset markers to the initial state."""
        self._markers = self._init_markers.copy()
        return

    def get_marked_structures(
        self, markers: Optional[numpy.typing.NDArray] = None
    ) -> list[Atoms]:
        """Return a list of Atoms for each marked position.

        Args:
            markers: Optional boolean or integer marker array. Uses
                ``self.markers`` if not provided.

        """
        if markers is None:
            curr_markers = self._markers
        else:
            curr_markers = np.asarray(markers)
            if curr_markers.dtype != bool:
                bool_markers = np.full(self._shape, False)
                for coord in curr_markers:
                    bool_markers[tuple(coord)] = True
                curr_markers = bool_markers

        marked_flat = np.flatnonzero(curr_markers)
        return [self._data[self._ind_map[f]] for f in marked_flat if f in self._ind_map]

    def tolist(self) -> list:
        """Reconstruct the nested list representation with None for empty cells."""
        data_1d = np.full(self._shape, None).flatten().tolist()
        for k, v in self._ind_map.items():
            data_1d[k] = self._data[v]

        return _reshape_data(data_1d, self._shape)

    def sel(self, **kwargs) -> Union["AtomsNDArray", Atoms]:
        """Select by coordinate label.

        Args:
            **kwargs: Dimension name to coordinate value mapping.
                Supports exact match (scalar) and slice-by-label.

        Returns:
            A new AtomsNDArray containing only the selected positions.
            If a single Atoms is selected, wraps it in a 1D array.

        Example:
            >>> arr.sel(temperature=300)
            >>> arr.sel(time=slice(0, 50))

        """
        if not kwargs:
            return self

        key: list[Union[int, slice]] = [slice(None)] * self.ndim
        for name, value in kwargs.items():
            if name not in self._dims:
                raise KeyError(f"Unknown dimension: {name}. Available dims: {self._dims}")
            if name not in self._coords:
                raise KeyError(
                    f"No coordinates for dimension '{name}'. "
                    f"Available coordinates: {list(self._coords.keys())}"
                )

            axis = self._dims.index(name)
            coord_labels = self._coords[name]

            if isinstance(value, slice):
                start = value.start
                stop = value.stop
                if start is not None:
                    start_idx = int(np.searchsorted(coord_labels, start, side="left"))
                else:
                    start_idx = None
                if stop is not None:
                    stop_idx = int(np.searchsorted(coord_labels, stop, side="right"))
                else:
                    stop_idx = None
                key[axis] = slice(start_idx, stop_idx, value.step)
            else:
                matches = np.where(coord_labels == value)[0]
                if len(matches) == 0:
                    raise KeyError(
                        f"Value {value} not found in coordinate '{name}'. "
                        f"Available: {coord_labels}"
                    )
                if len(matches) > 1:
                    raise KeyError(
                        f"Duplicate coordinate value {value} in dimension '{name}'."
                    )
                key[axis] = int(matches[0])

        result = self[tuple(key)]
        if isinstance(result, list):
            return AtomsNDArray(result)
        return result

    @property
    def loc(self) -> "_LocIndexer":
        """Label-based indexer.

        Provides pandas-like label-based access:
            >>> arr.loc["300K", :]
            >>> arr.loc["300K", "last"]

        For axes without coordinates, falls back to integer indexing.

        """
        return _LocIndexer(self)

    def take_last(self, axis: int = -1) -> Union[Atoms, list, None]:
        """Return the last non-None element along the given axis.

        Scans backward along the specified axis for each fixed position
        on the other axes, returning the last valid (non-None) Atoms.

        Args:
            axis: Axis along which to take the last valid element.
                Defaults to -1 (last axis).

        Returns:
            Single ``Atoms`` or ``None`` if ndim == 1 and no valid element.
            ``list[Atoms]`` (or ``list[list[...]]``) for higher dimensions.

        Example:
            >>> arr.take_last(axis=1)  # last frame of each trajectory
            >>> arr[0].take_last()     # single Atoms (first traj, last frame)

        """
        if axis < 0:
            axis += self.ndim
        if not 0 <= axis < self.ndim:
            raise IndexError(
                f"Axis {axis} out of bounds for {self.ndim}D array."
            )

        if self.ndim == 1:
            vals = np.flatnonzero(self._markers)
            if len(vals) == 0:
                return None
            return self._data[self._ind_map[vals[-1]]]

        other_axes = [i for i in range(self.ndim) if i != axis]
        result_shape = tuple(self._shape[i] for i in other_axes)

        result = []
        for combo in itertools.product(*[range(s) for s in result_shape]):
            # Build index with slice on target axis
            idx: list[Union[int, slice]] = [0] * self.ndim
            for oa, val in zip(other_axes, combo):
                idx[oa] = val
            idx[axis] = slice(None)

            markers_slice = self._markers[tuple(idx)]
            last = np.flatnonzero(markers_slice)
            if len(last) > 0:
                idx[axis] = int(last[-1])
                flat_idx = _map_idx(np.array(idx), self._shape)
                result.append(self._data[self._ind_map[flat_idx]])
            else:
                result.append(None)

        if not result_shape:
            return result[0] if result else None

        return _reshape_data(result, result_shape)

    @classmethod
    def _from_components(
        cls,
        shape: tuple[int, ...],
        data: list[Atoms],
        markers: numpy.typing.NDArray,
        ind_map: Mapping[int, int],
        dims: Optional[tuple[str, ...]] = None,
        coords: Optional[Mapping[str, np.ndarray]] = None,
    ) -> "AtomsNDArray":
        """Fast internal constructor (avoids re-processing)."""
        obj = cls.__new__(cls)
        obj._shape = shape
        obj._data = data
        obj._mask = markers.copy()
        obj._ind_map = dict(ind_map)
        obj._markers = markers.copy()
        obj._init_markers = markers.copy()
        obj._dims = dims if dims is not None else tuple(f"dim_{i}" for i in range(len(shape)))
        obj._coords = dict(coords) if coords is not None else {}
        return obj

    @classmethod
    def from_file(
        cls,
        target: Union[str, pathlib.Path, h5py.File],
        grp_name: str = "images",
    ) -> "AtomsNDArray":
        """Read an AtomsNDArray from an HDF5 file.

        Args:
            target: File path or open ``h5py.File``.
            grp_name: HDF5 group name where the array is stored.

        Returns:
            A new AtomsNDArray reconstructed from the file.

        """
        if isinstance(target, (str, pathlib.Path)):
            fopen = h5py.File(target, "r")
        else:
            assert isinstance(target, h5py.File)
            fopen = target

        try:
            grp = fopen.require_group(grp_name)  # type: ignore
            shape = tuple(grp.attrs["shape"])  # type: ignore
            images = cls._from_hd5grp(grp=grp)

            # Read markers — backward compat with multiple formats
            markers_data = np.array(grp["markers"][:])  # type: ignore
            if markers_data.dtype == bool:
                markers = markers_data
            elif markers_data.shape == shape:
                # Integer-encoded boolean (0/1) with full shape
                markers = markers_data.astype(bool)
            elif markers_data.ndim == 2 and markers_data.shape[1] == len(shape):
                # Old integer format (M, ndim) — convert to boolean
                markers = np.full(shape, False)
                for coord in markers_data:
                    markers[tuple(coord)] = True
            else:
                raise ValueError(
                    f"Unexpected markers shape {markers_data.shape} "
                    f"for array shape {shape}"
                )

            mapper_k = np.array(grp["map_k"][:])  # type: ignore
            mapper_v = np.array(grp["map_v"][:])  # type: ignore
            ind_map = dict(zip(mapper_k.tolist(), mapper_v.tolist()))

            # Read dims (optional — backward compat)
            dims = None
            if "dims" in grp.attrs:  # type: ignore
                dims = tuple(grp.attrs["dims"])  # type: ignore

            # Read coords (optional)
            coords = {}
            if "coord_names" in grp.attrs:  # type: ignore
                coord_names = list(grp.attrs["coord_names"])  # type: ignore
                for cname in coord_names:
                    if cname in grp:  # type: ignore
                        cdata = np.array(grp[cname][:])  # type: ignore
                        if cdata.dtype.kind == "S" or cdata.dtype == np.object_:  # type: ignore
                            cdata = np.array([s.decode() if isinstance(s, bytes) else str(s) for s in cdata])
                        coords[cname] = cdata

        except Exception:
            raise RuntimeError(traceback.format_exc())
        finally:
            if isinstance(target, (str, pathlib.Path)):
                fopen.close()

        return cls._from_components(
            shape=shape,
            data=images,
            markers=markers,
            ind_map=ind_map,
            dims=dims,
            coords=coords if coords else None,
        )

    @classmethod
    def _from_hd5grp(cls, grp) -> list[Atoms]:
        """Reconstruct Atoms objects from an HDF5 group."""
        natoms_list = grp["natoms"]

        images = []
        for natoms, box, pbc, atomic_numbers, tags, positions in zip(
            natoms_list,
            grp["box"],
            grp["pbc"],
            grp["atype"],
            grp["tags"],
            grp["positions"],
        ):
            atoms = Atoms(
                numbers=atomic_numbers[:natoms],
                positions=positions[:natoms, :],
                cell=box.reshape(3, 3),
                pbc=pbc,
                tags=tags[:natoms],
            )
            images.append(atoms)
        nimages = len(images)

        for name in RETAINED_INFO_NAMES:
            data = grp.get(name, default=None)
            if data is not None:
                for atoms, v in zip(images, data):
                    atoms.info[name] = v

        data = grp.get("momenta", default=None)
        if data is not None:
            for i, v in enumerate(data):
                a_v = v[: natoms_list[i]]
                if not np.all(np.isnan(a_v)):
                    images[i].set_momenta(a_v)

        results = [{} for _ in range(nimages)]
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

    def save_file(
        self,
        target: Union[str, pathlib.Path, h5py.File],
        grp_name: str = "images",
    ) -> None:
        """Write the AtomsNDArray to an HDF5 file.

        Args:
            target: File path or open ``h5py.File``.
            grp_name: HDF5 group name to write into.

        """
        if isinstance(target, (str, pathlib.Path)):
            fopen = h5py.File(target, "w")
        else:
            assert isinstance(target, h5py.File)
            fopen = target

        try:
            grp = fopen.create_group(grp_name)
            grp.attrs["shape"] = self._shape
            grp.attrs["dims"] = list(self._dims)

            # Write coordinate names and data
            if self._coords:
                grp.attrs["coord_names"] = list(self._coords.keys())
                for cname, cdata in self._coords.items():
                    if cdata.dtype.kind == "U":
                        cdata = [str(s) for s in cdata]
                        grp.create_dataset(cname, data=cdata, dtype=h5py.string_dtype())
                    else:
                        grp.create_dataset(cname, data=cdata)

            # Save structures
            self._convert_images(grp=grp, images=self._data)

            # Save markers (boolean)
            grp.create_dataset("markers", data=self._markers)

            # Save mapper
            mapper_k, mapper_v = [], []
            for k, v in self._ind_map.items():
                mapper_k.append(k)
                mapper_v.append(v)
            grp.create_dataset("map_k", data=mapper_k, dtype="i8")
            grp.create_dataset("map_v", data=mapper_v, dtype="i8")
        except Exception:
            print(f"{target=}  {grp_name=}")
            raise RuntimeError(traceback.format_exc())
        finally:
            if isinstance(target, (str, pathlib.Path)):
                fopen.close()

        return

    def _convert_images(self, grp, images: list[Atoms]) -> None:
        """Convert a flat list of Atoms objects to HDF5 datasets."""
        nimages = len(images)
        natoms_list = np.array([len(a) for a in images], dtype=np.int64)
        boxes = np.array([a.get_cell(complete=True) for a in images], dtype=np.float64).reshape(-1, 9)
        pbcs = np.array([a.get_pbc() for a in images], dtype=np.int8)
        max_natoms = max(natoms_list)
        atomic_numbers = np.zeros((nimages, max_natoms), dtype=np.int64)
        positions = np.zeros((nimages, max_natoms, 3), dtype=np.float64)
        tags = np.zeros((nimages, max_natoms), dtype=np.int64)
        for i, a in enumerate(images):
            atomic_numbers[i, : natoms_list[i]] = a.get_atomic_numbers()
            tags[i, : natoms_list[i]] = a.get_tags()
            positions[i, : natoms_list[i], :] = a.get_positions()

        _ = grp.create_dataset("natoms", data=natoms_list, dtype="i8")
        _ = grp.create_dataset("box", data=boxes, dtype="f8")
        _ = grp.create_dataset("pbc", data=pbcs, dtype="i8")
        _ = grp.create_dataset("atype", data=atomic_numbers, dtype="i8")
        _ = grp.create_dataset("tags", data=tags, dtype="i8")
        _ = grp.create_dataset("positions", data=positions, dtype="f8")

        for name, dtype in zip(RETAINED_INFO_NAMES, RETAINED_INFO_DTYPES):
            data = [a.info.get(name, np.nan) for a in images]
            if not np.all(np.isnan(data)):
                _ = grp.create_dataset(name, data=data, dtype=dtype)

        energies = np.array([a.get_potential_energy() for a in images], dtype=np.float64)
        free_energies = []
        for i, a in enumerate(images):
            try:
                free_energy = a.get_potential_energy(force_consistent=True)
            except Exception:
                free_energy = energies[i]
            free_energies.append(free_energy)

        forces = np.zeros((nimages, max(natoms_list), 3), dtype=np.float64)
        momenta = np.empty((nimages, max(natoms_list), 3), dtype=np.float64)
        momenta.fill(np.nan)
        for i, a in enumerate(images):
            forces[i, : natoms_list[i], :] = a.get_forces(apply_constraint=False)
            if "momenta" in a.arrays:
                momenta[i, : natoms_list[i], :] = a.get_momenta()

        _ = grp.create_dataset("energy", data=energies, dtype="f8")
        _ = grp.create_dataset("free_energy", data=free_energies, dtype="f8")
        _ = grp.create_dataset("forces", data=forces, dtype="f8")
        _ = grp.create_dataset("momenta", data=momenta, dtype="f8")

        return

    def __getitem__(self, key) -> Union[Atoms, list[Atoms]]:
        """"""
        if isinstance(key, (numbers.Integral, slice)):
            key = (key,) + (slice(None),) * (self.ndim - 1)
        elif not isinstance(key, tuple):
            raise IndexError("Index must be an integer, a slice or a tuple.")
        if len(key) > self.ndim:
            raise IndexError(f"{key} is out of dimension of {self._shape}.")

        indices, tshape = [], []
        for dim, i in enumerate(key):
            size = self._shape[dim]
            if isinstance(i, numbers.Integral):
                i = int(i)
                if i < -size or i >= size:
                    raise IndexError(
                        f"Index {i} is out of bounds for axis {dim} with size {size}."
                    )
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
                raise IndexError(f"Index must be an integer or a slice for dimension {dim}.")
            indices.append(curr_indices)

        products = np.array(list(itertools.product(*indices)))
        global_indices = [_map_idx(x, self._shape) for x in products]

        ret_data = []
        for x in global_indices:
            if x in self._ind_map:
                ret_data.append(self._data[self._ind_map[x]])
            else:
                ret_data.append(None)
        if tshape:
            ret = _reshape_data(ret_data, tuple(tshape))
        else:
            ret = ret_data[0]

        return ret

    def __len__(self) -> int:
        """"""
        return len(self._data)

    def __repr__(self) -> str:
        """"""
        return f"AtomsNDArray(nimages: {len(self)}, shape: {self.shape}, dims: {self._dims})"

    def __eq__(self, other) -> bool:
        """"""
        if not isinstance(other, AtomsNDArray):
            return NotImplemented
        if not np.array_equal(self._shape, other._shape):
            return False
        if self._dims != other._dims:
            return False
        if self._coords.keys() != other._coords.keys():
            return False
        for k in self._coords:
            if not np.array_equal(self._coords[k], other._coords[k]):
                return False
        if not np.array_equal(self._markers, other._markers):
            return False
        if len(self._data) != len(other._data):
            return False
        for a, b in zip(self._data, other._data):
            if a.get_chemical_symbols() != b.get_chemical_symbols():
                return False
            if not np.allclose(a.get_positions(), b.get_positions()):
                return False
            if not np.allclose(a.get_cell(complete=True), b.get_cell(complete=True)):
                return False
            if not np.array_equal(a.get_pbc(), b.get_pbc()):
                return False
        return True

    def __hash__(self) -> int:
        """"""
        return id(self)


class _LocIndexer:
    """Provides label-based indexing via ``arr.loc[key]``.

    For axes that have coordinates, all keys (including integers) are
    resolved against coordinate labels. For axes without coordinates,
    keys are passed through unchanged (positional indexing).

    Raises ``KeyError`` if a label is not found in the axis coordinates.

    """

    def __init__(self, array: AtomsNDArray):
        self._array = array

    def __getitem__(self, key) -> Union[Atoms, list[Atoms]]:
        if not isinstance(key, tuple):
            key = (key,)
        if len(key) > self._array.ndim:
            raise IndexError("Too many indices for AtomsNDArray.")

        key = key + (slice(None),) * (self._array.ndim - len(key))

        resolved = []
        for dim, k in zip(self._array.dims, key):
            if dim in self._array.coords:
                coord_labels = self._array.coords[dim]
                matches = np.where(coord_labels == k)[0]
                if len(matches) == 0:
                    raise KeyError(
                        f"Label {k!r} not found in coordinate '{dim}'. "
                        f"Available: {coord_labels}"
                    )
                if len(matches) > 1:
                    raise KeyError(
                        f"Duplicate label {k!r} in coordinate '{dim}'."
                    )
                resolved.append(int(matches[0]))
            else:
                resolved.append(k)

        return self._array[tuple(resolved)]
