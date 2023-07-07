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
                method = "property",
                properties = dict(
                    energy = dict(
                        range = [None, None],
                        nbins = 20,
                        sparsify = "filter"
                    )
                ),
            ),
            dict(
                method = "property",
                properties = dict(
                    energy = dict(
                        range = [-293., -292.],
                        nbins = 20,
                        sparsify = "hist"
                    )
                ),
                number = [4, 1.0],
                random_seed = 1112,
            )
        ]
    )

    return params


def test_props_1d(selection_params):
    """"""
    with tempfile.NamedTemporaryFile(suffix=".yaml") as tmp:
        with open(tmp.name, "w") as fopen:
            yaml.safe_dump(selection_params, fopen)
        
        with tempfile.TemporaryDirectory() as tmpdirname:
            #tmpdirname = "./xxx"
            _ = run_selection(tmp.name, "./r2.xyz", directory=tmpdirname)
            selected_frames = read(pathlib.Path(tmpdirname)/"selected_frames.xyz", ":")
    
    assert len(selected_frames) == 4

    #: selected_indices 2, 4, 54, 111
    t_energies = [-292.71506118, -292.47855862, -292.12845228, -292.01459094]
    #print(t_energies)

    energies = [a.get_potential_energy() for a in selected_frames]
    #print(energies)

    assert np.allclose(t_energies, energies)

    return


def test_props_2d(selection_params):
    """"""
    frames_ = read("./r2.xyz", ":")
    frames = []
    for i in range(2):
        frames.append(frames_[i*85:(i+1)*85])
    frames = AtomsNDArray(frames)

    selector = registers.create("variable", "selector", convert_name=True, **selection_params).value

    with tempfile.TemporaryDirectory() as tmpdirname:
        #tmpdirname = "./xxx"
        selector.directory = tmpdirname
        selected_frames = selector.select(frames)
    
    assert len(selected_frames) == 4

    #: selected_indices 2, 4, 54, 111
    t_energies = [-292.71506118, -292.47855862, -292.12845228, -292.01459094]
    #print(t_energies)

    energies = [a.get_potential_energy() for a in selected_frames]
    #print(energies)

    assert np.allclose(t_energies, energies)

    return


def test_props_2d_axis0(selection_params):
    """"""
    #print(selection_params)

    frames_ = read("./r2.xyz", ":")
    frames = []
    for i in range(2):
        frames.append(frames_[i*85:(i+1)*85])
    frames = AtomsNDArray(frames)

    selection_params["selection"][1]["axis"] = 0 # hist on axis 0
    selector = registers.create("variable", "selector", convert_name=True, **selection_params).value

    with tempfile.TemporaryDirectory() as tmpdirname:
        #tmpdirname = "./xxx"
        selector.directory = tmpdirname
        selected_frames = selector.select(frames)
    
    #write("./xxx.xyz", selected_frames)
    assert len(selected_frames) == 7

    #:
    t_energies = [
        -292.71506118, -292.47855862, -292.12845228, -292.27131595, 
        -292.99764801, -292.01459094, -292.14381737
    ]
    #print(t_energies)

    energies = [a.get_potential_energy() for a in selected_frames]
    #print(energies)

    assert np.allclose(t_energies, energies)

    return


def test_props_2dp(selection_params):
    """"""
    frames_ = read("./r2.xyz", ":")
    frames = []
    frames.append(frames_[:70])
    frames.append(frames_[70:])
    frames = AtomsNDArray(frames)

    selector = registers.create("variable", "selector", convert_name=True, **selection_params).value

    with tempfile.TemporaryDirectory() as tmpdirname:
        #tmpdirname = "./xxx"
        selector.directory = tmpdirname
        selected_frames = selector.select(frames)
    
    assert len(selected_frames) == 4

    #: selected_indices 2, 4, 54, 111
    t_energies = [-292.71506118, -292.47855862, -292.12845228, -292.01459094]
    #print(t_energies)

    energies = [a.get_potential_energy() for a in selected_frames]
    #print(energies)

    assert np.allclose(t_energies, energies)

    return


def test_props_2dp_axis0(selection_params):
    """"""
    #print(selection_params)

    frames_ = read("./r2.xyz", ":")
    frames = []
    frames.append(frames_[:70])
    frames.append(frames_[70:])
    frames = AtomsNDArray(frames)

    selection_params["selection"][1]["axis"] = 0 # hist on axis 0
    selector = registers.create("variable", "selector", convert_name=True, **selection_params).value

    with tempfile.TemporaryDirectory() as tmpdirname:
        #tmpdirname = "./xxx"
        selector.directory = tmpdirname
        selected_frames = selector.select(frames)
    
    #write("./xxx.xyz", selected_frames)
    assert len(selected_frames) == 7

    #:
    t_energies = [
        -292.71506118, -292.47855862, -292.12845228, -292.27131595, 
        -292.99764801, -292.01459094, -292.14381737
    ]
    #print(t_energies)

    energies = [a.get_potential_energy() for a in selected_frames]
    #print(energies)

    assert np.allclose(t_energies, energies)

    return


if __name__ == "__main__":
    ...