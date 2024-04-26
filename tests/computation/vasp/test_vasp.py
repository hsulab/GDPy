#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import List
import pathlib

import pytest
import tempfile
import yaml

import numpy as np

from ase import Atoms
from ase.io import read, write

from gdpx.cli.compute import run_worker, ComputerVariable


@pytest.fixture
def water_molecule() -> List[Atoms]:
    """"""
    # atoms = molecule("H2O", vacuum=10.)

    atoms = Atoms(
        numbers=[8, 1, 1],
        positions=[
            [10.00000000, 10.76320000, 10.59630000],
            [10.00000000, 11.52650000, 10.00000000],
            [10.00000000, 10.00000000, 10.00000000],
        ],
        cell=[[19.0, 0.0, 0.0], [0.0, 20.0, 0.0], [0.0, 0.0, 21.0]],
        pbc=False,
    )

    return [atoms]


@pytest.fixture
def structures() -> List[Atoms]:
    """H2"""

    atoms = Atoms(
        numbers=[1, 1],
        positions=[
            [10.00000000, 10.00000000, 10.72000000],
            [10.00000000, 10.00000000, 10.00000000],
        ],
        cell=[[9.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 11.0]],
        pbc=True,
    )

    return [atoms]


@pytest.mark.vasp
def test_vasp_spc(structures):
    """"""
    with open("./assets/vaspspc.yaml", "r") as fopen:
        vasp_params = yaml.safe_load(fopen)

    worker = ComputerVariable(**vasp_params).value[0]

    # - run
    with tempfile.TemporaryDirectory() as tmpdirname:
        worker.directory = tmpdirname
        # worker.directory = "./test_vasp_spc"
        worker.run(structures)
        worker.inspect(structures)
        if worker.get_number_of_running_jobs() == 0:
            results = worker.retrieve(include_retrieved=True)
        else:
            results = []

    final_energy = results[0][-1].get_potential_energy()

    assert np.allclose([final_energy], [-6.7066])


@pytest.mark.vasp
def test_vasp_md(structures):
    """"""
    with open("./assets/vaspmd.yaml", "r") as fopen:
        vasp_params = yaml.safe_load(fopen)

    # print(vasp_params)
    worker = ComputerVariable(**vasp_params).value[0]
    # print(worker)
    # print(f"{worker.driver.ignore_convergence =}")
    # print(f"{worker.driver.accept_bad_structure =}")
    # print(structures)

    # - run
    with tempfile.TemporaryDirectory() as tmpdirname:
        worker.directory = tmpdirname
        # worker.directory = "./test_vasp_md"
        worker.run(structures)
        worker.inspect(structures)
        if worker.get_number_of_running_jobs() == 0:
            results = worker.retrieve(include_retrieved=True)
        else:
            results = []

    # print(results[0][0].get_potential_energy())

    final_energy = results[0][-1].get_potential_energy()

    assert np.allclose([final_energy], [-6.6950])


if __name__ == "__main__":
    ...
