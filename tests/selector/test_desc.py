#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pathlib
import tempfile

import pytest
import yaml

import numpy as np

from ase.io import read, write

from GDPy.core.register import import_all_modules_for_register, registers
from GDPy.data.array import AtomsNDArray
from GDPy.selector.interface import run_selection


import_all_modules_for_register()


@pytest.fixture
def selection_params():
    """"""
    params = dict(
        selection = [
            dict(
                method = "descriptor",
                #mode: traj
                descriptor = dict(
                  name = "soap",
                  species = ["Al", "Cu", "O"],
                  rcut = 6.0,
                  nmax = 12,
                  lmax = 8,
                  sigma = 0.2,
                  average = "inner",
                  periodic = True
                ),
                sparsify = dict(
                    method = "fps",
                    min_distance = 0.1
                ),
                number = [4, 1.0],
                random_seed = 1112
            )
        ]
    )

    return params


def test_desc_1d(selection_params):
    """"""
    with tempfile.NamedTemporaryFile(suffix=".yaml") as tmp:
        with open(tmp.name, "w") as fopen:
            yaml.safe_dump(selection_params, fopen)
        
        with tempfile.TemporaryDirectory() as tmpdirname:
            _ = run_selection(tmp.name, "./r2.xyz", directory=tmpdirname)
            selected_frames = read(pathlib.Path(tmpdirname)/"selected_frames.xyz", ":")
    
    assert len(selected_frames) == 4

    #: fps start_index is 35, 52, 93, 156
    t_energies = [-285.47026343, -290.31314607, -291.51942674, -281.75524438]
    #print(t_energies)

    energies = [a.get_potential_energy() for a in selected_frames]
    #print(energies)

    assert np.allclose(t_energies, energies)

    return


def test_desc_2d(selection_params):
    """"""
    frames_ = read("./r2.xyz", ":")
    frames = []
    for i in range(2):
        frames.append(frames_[i*85:(i+1)*85])
    frames = AtomsNDArray(frames)

    selector = registers.create("variable", "selector", convert_name=True, **selection_params).value

    with tempfile.TemporaryDirectory() as tmpdirname:
        selector.directory = tmpdirname
        selected_frames = selector.select(frames)
    
    #write("./xxx.xyz", selected_frames)
    assert len(selected_frames) == 4

    #: fps start_index is 35, 52, 93, 156
    t_energies = [-285.47026343, -290.31314607, -291.51942674, -281.75524438]
    #print(t_energies)

    energies = [a.get_potential_energy() for a in selected_frames]
    #print(energies)

    assert np.allclose(t_energies, energies)

    return


def test_desc_2d_axis0(selection_params):
    """"""
    frames_ = read("./r2.xyz", ":")
    frames = []
    for i in range(2):
        frames.append(frames_[i*85:(i+1)*85])
    frames = AtomsNDArray(frames)

    selection_params["selection"][0]["axis"] = 0
    selector = registers.create("variable", "selector", convert_name=True, **selection_params).value

    with tempfile.TemporaryDirectory() as tmpdirname:
        #tmpdirname = "./xxx"
        selector.directory = tmpdirname
        selected_frames = selector.select(frames)
    
    #write("./xxx.xyz", selected_frames)
    assert len(selected_frames) == 8

    #:
    t_energies = [
        -284.09395726, -291.06722943, -290.31314607, -288.88948078, 
        -291.51942674, -286.45048884, -281.75524438, -289.39018868
    ]
    #print(t_energies)

    energies = [a.get_potential_energy() for a in selected_frames]
    #print(energies)

    assert np.allclose(t_energies, energies)

    return


def test_desc_2dp(selection_params):
    """"""
    frames_ = read("./r2.xyz", ":")
    frames = []
    frames.append(frames_[:70])
    frames.append(frames_[70:])
    frames = AtomsNDArray(frames)

    selector = registers.create("variable", "selector", convert_name=True, **selection_params).value

    with tempfile.TemporaryDirectory() as tmpdirname:
        selector.directory = tmpdirname
        selected_frames = selector.select(frames)
    
    #write("./xxx.xyz", selected_frames)
    assert len(selected_frames) == 4

    #: fps start_index is 35, 52, 93, 156
    t_energies = [-285.47026343, -290.31314607, -291.51942674, -281.75524438]
    #print(t_energies)

    energies = [a.get_potential_energy() for a in selected_frames]
    #print(energies)

    assert np.allclose(t_energies, energies)

    return


def test_desc_2dp_axis0(selection_params):
    """"""
    frames_ = read("./r2.xyz", ":")
    frames = []
    frames.append(frames_[:70])
    frames.append(frames_[70:])
    frames = AtomsNDArray(frames)

    selection_params["selection"][0]["axis"] = 0
    selector = registers.create("variable", "selector", convert_name=True, **selection_params).value

    with tempfile.TemporaryDirectory() as tmpdirname:
        #tmpdirname = "./xxx"
        selector.directory = tmpdirname
        selected_frames = selector.select(frames)
    
    #write("./xxx.xyz", selected_frames)
    assert len(selected_frames) == 8

    #:
    t_energies = [
        -285.76283507, -282.76850615, -291.06722943, -290.31314607, 
        -291.51942674, -286.86588241, -284.2378811, -289.39018868
    ]
    #print(t_energies)

    energies = [a.get_potential_energy() for a in selected_frames]
    #print(energies)

    assert np.allclose(t_energies, energies)

    return


if __name__ == "__main__":
    ...