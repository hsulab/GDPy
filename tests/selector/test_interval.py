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
                method = "interval",
                period = 37,
                include_first = True,
                include_last = True,
            )
        ]
    )

    return params


def test_intv_1d(selection_params):
    """"""
    with tempfile.NamedTemporaryFile(suffix=".yaml") as tmp:
        with open(tmp.name, "w") as fopen:
            yaml.safe_dump(selection_params, fopen)
        
        with tempfile.TemporaryDirectory() as tmpdirname:
            #tmpdirname = "./xxx"
            _ = run_selection(tmp.name, "./r2.xyz", directory=tmpdirname)
            selected_frames = read(pathlib.Path(tmpdirname)/"selected_frames.xyz", ":")
    
    assert len(selected_frames) == 6

    #: selected_indices is 0, 37, 74, 111, 148, 169
    t_energies = [
        -286.04976854, -290.86502869, -280.94478227, -292.01459094,
        -289.89265403, -284.85649391
    ]

    energies = [a.get_potential_energy() for a in selected_frames]

    assert np.allclose(t_energies, energies)

    return


def test_intv_2d(selection_params):
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
        markers = frames.markers
    #print(markers)
    
    assert len(selected_frames) == 6

    #:
    t_energies = [
        -286.04976854, -290.86502869, -280.94478227, -292.01459094,
        -289.89265403, -284.85649391
    ]

    energies = [a.get_potential_energy() for a in selected_frames]

    assert np.allclose(t_energies, energies)

    return


def test_intv_2d_axis0(selection_params):
    """"""
    #print(selection_params)

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
        -286.04976854, -290.86502869, -280.94478227, -283.21986666, 
        -292.27131595, -286.86588241, -285.31668644, -284.85649391
    ]

    energies = [a.get_potential_energy() for a in selected_frames]

    assert np.allclose(t_energies, energies)

    return


def test_intv_2dp(selection_params):
    """"""
    frames_ = read("./r2.xyz", ":")
    frames = []
    frames.append(frames_[0:70])
    frames.append(frames_[70:])
    frames = AtomsNDArray(frames)

    selector = registers.create("variable", "selector", convert_name=True, **selection_params).value

    with tempfile.TemporaryDirectory() as tmpdirname:
        #tmpdirname = "./xxx"
        selector.directory = tmpdirname
        selected_frames = selector.select(frames)
        markers = frames.markers
    #print(markers)
    
    assert len(selected_frames) == 6

    #:
    t_energies = [
        -286.04976854, -290.86502869, -280.94478227, -292.01459094,
        -289.89265403, -284.85649391
    ]

    energies = [a.get_potential_energy() for a in selected_frames]

    assert np.allclose(t_energies, energies)

    return

def test_intv_2dp_axis0(selection_params):
    """"""
    #print(selection_params)

    frames_ = read("./r2.xyz", ":")
    frames = []
    frames.append(frames_[0:70])
    frames.append(frames_[70:])
    frames = AtomsNDArray(frames)

    selection_params["selection"][0]["axis"] = 0
    selector = registers.create("variable", "selector", convert_name=True, **selection_params).value

    with tempfile.TemporaryDirectory() as tmpdirname:
        #tmpdirname = "./xxx"
        selector.directory = tmpdirname
        selected_frames = selector.select(frames)
        markers = frames.markers
    #print(markers)
    
    # 0 37 69
    # 0 37 74 99
    assert len(selected_frames) == 7

    #:
    t_energies = [
        -286.04976854, -290.86502869, -290.91516215, -283.53940274,
        -281.73179116, -287.16805319, -284.85649391
    ]

    energies = [a.get_potential_energy() for a in selected_frames]

    assert np.allclose(t_energies, energies)

    return


if __name__ == "__main__":
    ...