#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import copy
import tempfile

import pytest

import numpy as np

from ase.io import read, write


from gdpx.factory.computer import create_workers
from gdpx.utils.parser import parse_input_file


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


@pytest.mark.lammps
def test_reax_spc():
    """"""
    atoms = read("./assets/Pd38_oct.xyz")
    structures = [atoms]

    worker = create_workers(parse_input_file("./assets/reaxspc.yaml"))[0]

    results = run_computation(structures, worker)

    energy = results[-1][-1].get_potential_energy()

    assert np.allclose(energy, -114.443082558)


@pytest.mark.lammps
def test_reax_min():
    """"""
    atoms = read("./assets/Pd38_oct.xyz")
    structures = [atoms]

    worker = create_workers(parse_input_file("./assets/reaxmin.yaml"))[0]

    results = run_computation(structures, worker)

    energy = results[-1][-1].get_potential_energy()

    assert np.allclose(energy, -116.122977589)


@pytest.mark.lammps
def test_reax_nvt():
    """"""
    atoms = read("./assets/Pd38_oct.xyz")
    structures = [atoms]

    worker = create_workers(parse_input_file("./assets/reaxmd.yaml"))[0]

    results = run_computation(structures, worker)

    energy = results[-1][-1].get_potential_energy()

    assert np.allclose(energy, -115.205969893)


if __name__ == "__main__":
    ...
