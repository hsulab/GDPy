#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pathlib
import tempfile

import pytest

import numpy as np

from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read, write

from gdpx.data.array import AtomsNDArray


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def aa3d():
    """"""
    frames = read("./bands.xyz", ":")

    bands = []
    for i in range(3):
        bands.append(frames[i*7:(i+1)*7])

    return AtomsNDArray([bands]) # shape (1, 3, 7)

@pytest.fixture
def aa3d_pad():
    """"""
    frames = read("./bands.xyz", ":")

    bands = []
    bands.append(frames[0:9])    # 9
    bands.append(frames[9:14])   # 5
    bands.append(frames[14:21])  # 7

    bands2 = []
    bands2.append(frames[0:6])    # 6
    bands2.append(frames[6:13])   # 7
    bands2.append(frames[13:16])  # 3
    bands2.append(frames[16:21])  # 5

    return AtomsNDArray([bands, bands2]) # shape (2, 4?, 9?)

@pytest.fixture
def random_atoms():
    """Create artificial atoms with random positions and known ids."""
    rng = np.random.default_rng(42)
    atoms_pool = []
    for i in range(50):
        a = Atoms("H", positions=rng.random((1, 3)) * 10, cell=[10, 10, 10], pbc=True)
        a.info["id"] = i
        a.calc = SinglePointCalculator(a, energy=float(i), forces=rng.random((1, 3)))
        atoms_pool.append(a)
    return atoms_pool


# ---------------------------------------------------------------------------
# Existing tests (unchanged)
# ---------------------------------------------------------------------------

def test_shape(aa3d):
    """"""
    assert len(aa3d) == 21
    assert aa3d.shape == (1, 3, 7)

    return

def test_read_and_save(aa3d):
    """"""
    # - test shape
    aa3d.markers = [[0,0,3], [0,1,2]]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".h5") as tmp:
        aa3d.save_file(tmp.name)
        new_aa3d = AtomsNDArray.from_file(tmp.name)

    assert aa3d.shape == (1,3,7,)
    assert new_aa3d.shape == (1,3,7,)
    assert np.all(aa3d.markers == new_aa3d.markers)
    assert aa3d._ind_map == new_aa3d._ind_map

    return

def test_markers(aa3d):
    """"""
    # - set new makers
    markers = [
        [0, 1, 2], [0, 1, 3]
    ]
    aa3d.markers = markers
    marked_images = aa3d.get_marked_structures()
    ranks = [a.info["rank"] for a in marked_images]

    assert ranks == [9, 10]

    return

def test_padded_array(aa3d_pad):
    """"""
    a33dp = aa3d_pad

    assert a33dp.shape == (2, 4, 9)

    return

def test_read_and_save_pad(aa3d_pad):
    """"""
    aa3d = aa3d_pad

    # - test shape
    aa3d.markers = [[0,0,3], [0,1,2]]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".h5") as tmp:
        aa3d.save_file(tmp.name)
        new_aa3d = AtomsNDArray.from_file(tmp.name)

    assert aa3d.shape == (2,4,9,)
    assert new_aa3d.shape == (2,4,9,)
    assert np.all(aa3d.markers == new_aa3d.markers)
    assert aa3d._ind_map == new_aa3d._ind_map

    return


# ---------------------------------------------------------------------------
# __getitem__ tests for 1D, 2D, 3D
# ---------------------------------------------------------------------------

def test_getitem_1d(random_atoms):
    """1D array: integer, slice, negative indexing."""
    arr = AtomsNDArray(random_atoms[:10])
    assert arr.shape == (10,)

    # Integer indexing
    assert arr[0].info["id"] == 0
    assert arr[5].info["id"] == 5
    assert arr[-1].info["id"] == 9
    assert arr[-3].info["id"] == 7

    # Slice indexing
    s = arr[:3]
    assert len(s) == 3
    assert [a.info["id"] for a in s] == [0, 1, 2]

    s = arr[3:6]
    assert [a.info["id"] for a in s] == [3, 4, 5]

    s = arr[::2]
    assert [a.info["id"] for a in s] == [0, 2, 4, 6, 8]

    s = arr[::-1]
    assert s[0].info["id"] == 9


def test_getitem_2d(random_atoms):
    """2D array: single Atoms, slices, negative indices."""
    data = [random_atoms[i*5:(i+1)*5] for i in range(4)]
    arr = AtomsNDArray(data)
    assert arr.shape == (4, 5)

    # Single element
    assert arr[0, 0].info["id"] == 0
    assert arr[2, 3].info["id"] == 2*5 + 3
    assert arr[-1, -1].info["id"] == 3*5 + 4

    # Slice on each axis
    col = arr[:, 0]
    assert len(col) == 4
    assert [a.info["id"] for a in col] == [0, 5, 10, 15]

    row = arr[1, :]
    assert len(row) == 5
    assert [a.info["id"] for a in row] == [5, 6, 7, 8, 9]

    # Negative slice
    last_col = arr[:, -1]
    assert [a.info["id"] for a in last_col] == [4, 9, 14, 19]


def test_getitem_3d(random_atoms):
    """3D array: shape (2, 3, 4)."""
    data = []
    for i in range(2):
        planes = []
        for j in range(3):
            planes.append(random_atoms[(i*3 + j)*4 : (i*3 + j + 1)*4])
        data.append(planes)
    arr = AtomsNDArray(data)
    assert arr.shape == (2, 3, 4)

    # Single element
    assert arr[0, 1, 2].info["id"] == (0*3 + 1)*4 + 2
    assert arr[1, 0, 3].info["id"] == (1*3 + 0)*4 + 3

    # Full planes via slices
    plane = arr[0, :, :]
    assert len(plane) == 3 and len(plane[0]) == 4
    assert plane[1][2].info["id"] == arr[0, 1, 2].info["id"]

    # Slice on middle axis
    slab = arr[:, 0, :]
    assert len(slab) == 2 and len(slab[0]) == 4


# ---------------------------------------------------------------------------
# __getitem__ with None / padded cells
# ---------------------------------------------------------------------------

def test_getitem_none_in_1d(random_atoms):
    """1D array with None — indexing returns None for empty cells."""
    data = [random_atoms[0], None, random_atoms[1], None, random_atoms[2]]
    arr = AtomsNDArray([data])
    assert arr.shape == (1, 5)
    assert arr[0, 0].info["id"] == 0
    assert arr[0, 1] is None
    assert arr[0, 2].info["id"] == 1
    assert arr[0, 3] is None
    assert arr[0, 4].info["id"] == 2


def test_getitem_none_in_2d(random_atoms):
    """2D ragged array — padded positions return None."""
    data = [
        [random_atoms[0], random_atoms[1], random_atoms[2], None, None],
        [random_atoms[3], None, None, None, None],
        [random_atoms[4], random_atoms[5], random_atoms[6], random_atoms[7], random_atoms[8]],
    ]
    arr = AtomsNDArray(data)
    assert arr.shape == (3, 5)

    # Valid positions
    assert arr[0, 0].info["id"] == 0
    assert arr[0, 2].info["id"] == 2
    assert arr[2, 4].info["id"] == 8

    # None positions
    assert arr[0, 3] is None
    assert arr[0, 4] is None
    assert arr[1, 1] is None
    assert arr[1, 2] is None
    assert arr[2, 0].info["id"] == 4  # not None

    # Slice with None
    row = arr[0, :]
    assert [a is not None for a in row] == [True, True, True, False, False]

    row = arr[1, :]
    assert [a is not None for a in row] == [True, False, False, False, False]

    # Last column: some may be None
    last_col = arr[:, -1]
    assert [None if a is None else a.info["id"] for a in last_col] == [None, None, 8]

    # Sub-array via double slice
    sub = arr[0:2, 0:3]
    assert len(sub) == 2 and len(sub[0]) == 3
    assert sub[0][0].info["id"] == 0
    assert sub[1][1] is None


# ---------------------------------------------------------------------------
# take_last tests
# ---------------------------------------------------------------------------

def test_take_last_1d(random_atoms):
    """take_last on 1D array returns last non-None."""
    data = [random_atoms[0], None, random_atoms[1], None, random_atoms[2]]
    arr = AtomsNDArray(data)

    last = arr.take_last()
    assert last is not None
    assert last.info["id"] == 2

    # take_last on the 1D atom list (wrapped as 2D for ragged)
    arr2 = AtomsNDArray([data])
    last2 = arr2.take_last(axis=1)
    assert isinstance(last2, list)
    assert len(last2) == 1
    assert last2[0].info["id"] == 2


def test_take_last_1d_all_none(random_atoms):
    """take_last on 1D with no valid cells returns None."""
    arr = AtomsNDArray([None, None, None])
    last = arr.take_last()
    assert last is None


def test_take_last_2d_ragged(random_atoms):
    """take_last on ragged 2D returns last non-None per row."""
    data = [
        [random_atoms[0], random_atoms[1], random_atoms[2], None, None],
        [random_atoms[3], None, None, None, None],
        [random_atoms[4], random_atoms[5], random_atoms[6], random_atoms[7], random_atoms[8]],
    ]
    arr = AtomsNDArray(data)

    last = arr.take_last(axis=1)
    assert len(last) == 3
    assert last[0].info["id"] == 2   # last non-None in row 0
    assert last[1].info["id"] == 3   # last non-None in row 1
    assert last[2].info["id"] == 8   # last non-None in row 2

    # Compare with [:,-1] which may return None
    assert arr[:, -1][0] is None     # position -1 is None
    assert arr[:, -1][1] is None     # position -1 is None
    assert arr[:, -1][2].info["id"] == 8  # position -1 is valid


def test_take_last_2d_axis0(random_atoms):
    """take_last along axis 0 selects last row with data."""
    data = [
        [random_atoms[0], random_atoms[1], None],
        [random_atoms[2], None, None],
        [random_atoms[3], random_atoms[4], random_atoms[5]],
    ]
    arr = AtomsNDArray(data)

    last = arr.take_last(axis=0)
    assert len(last) == 3  # shape (3,) — column-wise
    # Column 0: valid at [0], [1], [2] -> last is [2] (atoms[3])
    assert last[0].info["id"] == 3
    # Column 1: valid at [0], [2] -> last is [2] (atoms[4])
    assert last[1].info["id"] == 4
    # Column 2: valid at [2] -> last is [2] (atoms[5])
    assert last[2].info["id"] == 5


def test_take_last_3d(random_atoms):
    """take_last on 3D reduces ndim by 1."""
    data = []
    for i in range(2):
        planes = []
        for j in range(3):
            planes.append(random_atoms[(i*3 + j)*4 : (i*3 + j + 1)*4])
        data.append(planes)
    arr = AtomsNDArray(data)
    assert arr.shape == (2, 3, 4)

    # take_last along axis = 1 -> shape (2, 4)
    tl = arr.take_last(axis=1)
    assert len(tl) == 2
    assert len(tl[0]) == 4
    # Last valid along axis 1 for [0, :, k] is position 2
    assert tl[0][0].info["id"] == arr[0, 2, 0].info["id"]

    # take_last along axis = 2 -> shape (2, 3)
    tl = arr.take_last(axis=2)
    assert len(tl) == 2
    assert len(tl[0]) == 3
    assert tl[0][1].info["id"] == arr[0, 1, 3].info["id"]


# ---------------------------------------------------------------------------
# sel / loc tests
# ---------------------------------------------------------------------------

def test_sel_exact_match(random_atoms):
    """sel() with a single coordinate value."""
    data = [random_atoms[:5], random_atoms[5:10]]
    arr = AtomsNDArray(data, dims=("temp", "time"), coords={"temp": [300, 500]})

    selected = arr.sel(temp=300)
    assert selected.shape == (5,)
    ids = [a.info["id"] for a in selected]
    assert ids == [0, 1, 2, 3, 4]


def test_sel_slice(random_atoms):
    """sel() with a slice by coordinate label."""
    data = [random_atoms[:5], random_atoms[5:10], random_atoms[10:15]]
    arr = AtomsNDArray(
        data, dims=("temp", "time"), coords={"temp": [300, 400, 500]}
    )

    selected = arr.sel(temp=slice(400, 500))
    assert selected.shape == (2, 5)


def test_sel_unknown_dim(random_atoms):
    """sel() raises KeyError for unknown dimension."""
    data = [random_atoms[:5], random_atoms[5:10]]
    arr = AtomsNDArray(data, dims=("temp", "time"), coords={"temp": [300, 500]})

    with pytest.raises(KeyError, match="Unknown dimension"):
        arr.sel(pressure=1.0)


def test_sel_missing_label(random_atoms):
    """sel() raises KeyError for label not in coords."""
    data = [random_atoms[:5], random_atoms[5:10]]
    arr = AtomsNDArray(data, dims=("temp", "time"), coords={"temp": [300, 500]})

    with pytest.raises(KeyError, match="not found"):
        arr.sel(temp=999)


def test_sel_duplicate_label(random_atoms):
    """sel() raises KeyError for duplicate coordinate values."""
    data = random_atoms[:3]
    arr = AtomsNDArray(data, dims=("temp",), coords={"temp": [300, 400, 400]})
    with pytest.raises(KeyError, match="Duplicate coordinate"):
        arr.sel(temp=400)


def test_loc_label_resolution(random_atoms):
    """loc[] resolves integer keys as coordinate labels."""
    data = [random_atoms[:5], random_atoms[5:10]]
    arr = AtomsNDArray(data, dims=("temp", "time"), coords={"temp": [300, 500]})

    selected = arr.loc[300, :]
    assert len(selected) == 5
    assert selected[0].info["id"] == 0

    # loc with string keys
    arr2 = AtomsNDArray(
        data, dims=("run", "step"), coords={"run": ["A", "B"]}
    )
    selected2 = arr2.loc["A", :]
    assert len(selected2) == 5
    assert selected2[0].info["id"] == 0


def test_loc_no_coords(random_atoms):
    """loc[] falls through to int indexing for axes without coords."""
    data = [random_atoms[:5], random_atoms[5:10]]
    arr = AtomsNDArray(data)

    selected = arr.loc[0, :]
    assert len(selected) == 5
    assert selected[0].info["id"] == 0


def test_loc_missing_label(random_atoms):
    """loc[] raises KeyError for missing label."""
    data = [random_atoms[:5], random_atoms[5:10]]
    arr = AtomsNDArray(data, dims=("temp", "time"), coords={"temp": [300, 500]})

    with pytest.raises(KeyError, match="not found"):
        arr.loc[999]


# ---------------------------------------------------------------------------
# Marker tests
# ---------------------------------------------------------------------------

def test_markers_default_is_boolean(random_atoms):
    """Default markers are boolean with True for all data cells."""
    data = [random_atoms[:3], random_atoms[3:6]]
    arr = AtomsNDArray(data)

    assert arr.markers.dtype == bool
    assert arr.markers.shape == arr.shape
    assert np.all(arr.markers)


def test_markers_none_positions(random_atoms):
    """Markers are False at None (padded) positions."""
    data = [
        [random_atoms[0], random_atoms[1], None],
        [random_atoms[2], random_atoms[3], random_atoms[4]],
    ]
    arr = AtomsNDArray(data)

    assert arr.markers[0, 0] == True
    assert arr.markers[0, 1] == True
    assert arr.markers[0, 2] == False  # None position
    assert arr.markers[1, 0] == True
    assert arr.markers[1, 1] == True
    assert arr.markers[1, 2] == True


def test_markers_set_boolean(random_atoms):
    """Setting markers with a boolean array."""
    data = [random_atoms[:5], random_atoms[5:10]]
    arr = AtomsNDArray(data)

    new_markers = np.full(arr.shape, False)
    new_markers[0, :] = True  # only first row
    arr.markers = new_markers

    assert np.count_nonzero(arr.markers) == 5
    ids = [a.info["id"] for a in arr.get_marked_structures()]
    assert ids == [0, 1, 2, 3, 4]


def test_markers_set_integer_old_format(random_atoms):
    """Setting markers with old (M, ndim) integer format auto-converts."""
    data = [random_atoms[:5], random_atoms[5:10]]
    arr = AtomsNDArray(data)

    arr.markers = [[0, 1], [0, 3], [1, 2]]
    assert arr.markers.dtype == bool
    assert np.count_nonzero(arr.markers) == 3

    ids = [a.info["id"] for a in arr.get_marked_structures()]
    assert ids == [1, 3, 7]


def test_markers_reset(random_atoms):
    """reset_markers restores init_markers."""
    data = [random_atoms[:5], random_atoms[5:10]]
    arr = AtomsNDArray(data)

    initial_count = np.count_nonzero(arr.markers)
    arr.markers = [[0, 0], [0, 1]]
    assert np.count_nonzero(arr.markers) == 2

    arr.reset_markers()
    assert np.count_nonzero(arr.markers) == initial_count
    assert np.array_equal(arr.markers, arr.init_markers)


def test_markers_get_marked_structures_with_args(random_atoms):
    """get_marked_structures accepts integer or boolean markers."""
    data = [random_atoms[:5], random_atoms[5:10]]
    arr = AtomsNDArray(data)

    # Integer format
    ids_int = [a.info["id"] for a in arr.get_marked_structures([[0, 0], [1, 1]])]
    assert ids_int == [0, 6]

    # Boolean format
    bool_mask = np.full(arr.shape, False)
    bool_mask[0, 2] = True
    bool_mask[1, 3] = True
    ids_bool = [a.info["id"] for a in arr.get_marked_structures(bool_mask)]
    assert ids_bool == [2, 8]

    # None (default) → uses self.markers
    ids_default = [a.info["id"] for a in arr.get_marked_structures()]
    assert len(ids_default) == 10


def test_markers_count_nonzero(random_atoms):
    """Use count_nonzero instead of len() for boolean markers."""
    data = [[random_atoms[0], None], [random_atoms[1], random_atoms[2]]]
    arr = AtomsNDArray(data)

    assert int(np.count_nonzero(arr.markers)) == 3
    assert int(np.count_nonzero(arr.init_markers)) == 3


# ---------------------------------------------------------------------------
# HDF5 roundtrip tests
# ---------------------------------------------------------------------------

def test_hdf5_roundtrip_dims_coords(random_atoms):
    """HDF5 roundtrip preserves dims and coords."""
    data = [random_atoms[:5], random_atoms[5:10]]
    arr = AtomsNDArray(
        data, dims=("temp", "time"), coords={"temp": [300, 500]}
    )

    with tempfile.NamedTemporaryFile(suffix=".h5") as f:
        arr.save_file(f.name)
        loaded = AtomsNDArray.from_file(f.name)

    assert loaded.dims == ("temp", "time")
    assert np.array_equal(loaded.coords["temp"], arr.coords["temp"])
    assert loaded == arr


def test_hdf5_roundtrip_string_coords(random_atoms):
    """String coordinate labels roundtrip correctly."""
    data = random_atoms[:5]
    arr = AtomsNDArray(data, dims=("run",), coords={"run": ["A", "B", "C", "D", "E"]})

    with tempfile.NamedTemporaryFile(suffix=".h5") as f:
        arr.save_file(f.name)
        loaded = AtomsNDArray.from_file(f.name)

    assert np.array_equal(loaded.coords["run"], np.array(["A", "B", "C", "D", "E"]))


def test_hdf5_roundtrip_markers_boolean(random_atoms):
    """HDF5 roundtrip preserves boolean markers."""
    data = [random_atoms[:5], random_atoms[5:10]]
    arr = AtomsNDArray(data)
    arr.markers = [[0, 1], [0, 3], [1, 2]]

    with tempfile.NamedTemporaryFile(suffix=".h5") as f:
        arr.save_file(f.name)
        loaded = AtomsNDArray.from_file(f.name)

    assert loaded.markers.dtype == bool
    assert np.array_equal(loaded.markers, arr.markers)
    assert np.count_nonzero(loaded.markers) == 3


def test_hdf5_roundtrip_ragged(random_atoms):
    """HDF5 roundtrip preserves padded structure."""
    data = [
        [random_atoms[0], random_atoms[1], None, None],
        [random_atoms[2], None, None, None],
        [random_atoms[3], random_atoms[4], random_atoms[5], random_atoms[6]],
    ]
    arr = AtomsNDArray(data)
    assert arr.shape == (3, 4)

    with tempfile.NamedTemporaryFile(suffix=".h5") as f:
        arr.save_file(f.name)
        loaded = AtomsNDArray.from_file(f.name)

    assert loaded.shape == (3, 4)
    assert np.array_equal(loaded.markers, arr.markers)
    assert loaded[0, 2] is None
    assert loaded[1, 1] is None
    assert loaded[2, 3] is not None


def test_hdf5_roundtrip_3d(random_atoms):
    """HDF5 roundtrip for 3D array."""
    data = []
    for i in range(2):
        planes = []
        for j in range(3):
            planes.append(random_atoms[(i*3 + j)*4 : (i*3 + j + 1)*4])
        data.append(planes)
    arr = AtomsNDArray(data)

    with tempfile.NamedTemporaryFile(suffix=".h5") as f:
        arr.save_file(f.name)
        loaded = AtomsNDArray.from_file(f.name)

    assert loaded.shape == (2, 3, 4)


# ---------------------------------------------------------------------------
# dims / coords / repr tests
# ---------------------------------------------------------------------------

def test_default_dims(random_atoms):
    """Default dims are (dim_0, dim_1, ...)."""
    arr = AtomsNDArray([random_atoms[:3], random_atoms[3:6]])
    assert arr.dims == ("dim_0", "dim_1")

    arr2 = AtomsNDArray(random_atoms[:5])
    assert arr2.dims == ("dim_0",)


def test_repr(random_atoms):
    """__repr__ includes shape and dims."""
    arr = AtomsNDArray(
        [random_atoms[:5], random_atoms[5:10]],
        dims=("temp", "time"),
    )
    r = repr(arr)
    assert "AtomsNDArray" in r
    assert "temp" in r
    assert "time" in r
    assert "(2, 5)" in r


def test_eq_same(random_atoms):
    """Identical arrays compare equal."""
    a = AtomsNDArray([random_atoms[:3], random_atoms[3:6]])
    b = AtomsNDArray([random_atoms[:3], random_atoms[3:6]])
    assert a == b


def test_eq_different_shape(random_atoms):
    """Different shapes compare not equal."""
    a = AtomsNDArray([random_atoms[:3], random_atoms[3:6]])
    b = AtomsNDArray(random_atoms[:6])
    assert a != b


def test_eq_different_markers(random_atoms):
    """Different markers compare not equal."""
    a = AtomsNDArray([random_atoms[:3], random_atoms[3:6]])
    b = AtomsNDArray([random_atoms[:3], random_atoms[3:6]])
    b.markers = [[0, 0], [0, 1]]
    assert a != b


# ---------------------------------------------------------------------------
# Construction from AtomsNDArray (copy)
# ---------------------------------------------------------------------------

def test_copy_constructor(random_atoms):
    """Constructing from another AtomsNDArray copies data and markers."""
    a = AtomsNDArray(
        [random_atoms[:5], random_atoms[5:10]],
        dims=("temp", "time"),
        coords={"temp": [300, 500]},
    )
    a.markers = [[0, 1], [0, 3]]
    b = AtomsNDArray(a)

    assert b.shape == a.shape
    assert b.dims == a.dims
    assert np.array_equal(b.coords["temp"], a.coords["temp"])
    assert np.array_equal(b.markers, a.markers)
    assert b == a


# ---------------------------------------------------------------------------
# ndim property
# ---------------------------------------------------------------------------

def test_ndim(random_atoms):
    """ndim property matches len(shape)."""
    assert AtomsNDArray(random_atoms[:5]).ndim == 1
    assert AtomsNDArray([random_atoms[:3], random_atoms[3:6]]).ndim == 2
    assert AtomsNDArray([[[random_atoms[0]]]]).ndim == 3


# ---------------------------------------------------------------------------
# Invalid inputs
# ---------------------------------------------------------------------------

def test_empty_data_raises():
    """Empty data raises ValueError."""
    with pytest.raises(ValueError):
        AtomsNDArray([])


def test_invalid_data_type():
    """Non-list, non-AtomsNDArray raises TypeError."""
    with pytest.raises(TypeError):
        AtomsNDArray(data="invalid")  # type: ignore


if __name__ == "__main__":
    ...
