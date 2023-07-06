#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tempfile

import pytest

from ase import Atoms
from ase.io import read, write

from GDPy.data.array import AtomsNDArray


@pytest.fixture
def aa3d():
    """"""
    frames = read("./bands.xyz", ":")
    #frames = [Atoms() for _ in range(21)]
    #for i, atoms in enumerate(frames):
    #    atoms.info["rank"] = i

    bands = []
    for i in range(3):
        bands.append(frames[i*7:(i+1)*7])

    return AtomsNDArray([bands]) # shape (1, 3, 7)


def test_shape(aa3d):
    """"""
    aa = aa3d

    assert len(aa3d) == 21
    assert aa3d.shape == (1, 3, 7)

    return

def test_read_and_save(aa3d):
    """"""
    # - test shape
    aa3d.markers = [[0,0,3], [0,1,2]]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".h5") as tmp:
        # tmp = pathlib.Path("./d.h5")
        aa3d.save_file(tmp.name)
        new_aa3d = AtomsNDArray.from_file(tmp.name)
    #print(aa3d.markers)
    #print(new_aa3d.markers)

    assert aa3d.shape == (1,3,7,)
    assert new_aa3d.shape == (1,3,7,)

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


if __name__ == "__main__":
    ...
