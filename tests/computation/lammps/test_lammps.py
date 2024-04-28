#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import copy
import os
import pathlib
import shutil
import tempfile

import yaml
import pytest

import numpy as np

from ase.io import read, write

from gdpx import config
from gdpx.cli.compute import convert_config_to_potter

"""
LAMMPS 23 Jun 2022 - Update 1 (where test values are from)
LAMMPS 2  Aug 2023 (DP version may give different energy numbers)
"""


def run_computation(structures, worker):
    """"""
    with tempfile.TemporaryDirectory() as tmpdirname:
        worker.directory = tmpdirname
        # worker.directory = "./_xxx"
        worker.run(structures)
        worker.inspect(structures)
        if worker.get_number_of_running_jobs() == 0:
            results = worker.retrieve(include_retrieved=True)
        else:
            results = []

    return results


def test_reax_nvt():
    """"""
    atoms = read("./assets/Pd38_oct.xyz")
    structures = [atoms]

    worker = convert_config_to_potter("./assets/reaxnvt.yaml")[0]

    results = run_computation(structures, worker)

    num_frames = len(results[-1])
    assert num_frames == 18

    energy = results[-1][-1].get_potential_energy()

    assert np.allclose(energy, -115.088791411)

    return


def test_reax_nvt_continue():
    """"""
    atoms = read("./assets/Pd38_oct.xyz")
    structures = [atoms]

    with open("./assets/reaxnvt.yaml", "r") as fopen:
        worker_params = yaml.safe_load(fopen)

    with tempfile.TemporaryDirectory() as tmpdirname:
        wdir = tmpdirname
        wdir = "./_xxx"
        wdir = pathlib.Path(wdir)
        # run first simulation
        worker = convert_config_to_potter(worker_params)[0]
        worker.directory = wdir
        worker.run(structures)
        # worker.inspect(structures)
        # if worker.get_number_of_running_jobs() == 0:
        #     results = worker.retrieve(include_retrieved=True)
        # else:
        #     results = []

        # run more steps
        os.remove(wdir / "_local_jobs.json")
        worker_params["driver"]["run"]["steps"] = 29

        worker = convert_config_to_potter(worker_params)[0]
        worker.directory = wdir
        worker.run(structures)
        worker.inspect(structures)
        if worker.get_number_of_running_jobs() == 0:
            results = worker.retrieve(include_retrieved=True)
        else:
            results = []

    energy = results[-1][-1].get_potential_energy()

    assert energy == pytest.approx(-116.183878136)

    return


if __name__ == "__main__":
    ...
