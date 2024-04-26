#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import tempfile

import pytest
import yaml

import numpy as np

from ase.io import read, write

from gdpx.cli.compute import convert_config_to_potter


@pytest.mark.vasp_rxn
def test_vasp_neb():
    """"""
    structures = read("./assets/CO+O_mep.xyz", ":")

    with open("./assets/vaspneb.yaml", "r") as fopen:
        vasp_params = yaml.safe_load(fopen)

    worker = convert_config_to_potter(vasp_params)
    print(f"{worker =}")

    with tempfile.TemporaryDirectory() as tmpdirname:
        worker.directory = tmpdirname
        # worker.directory = "./test_vasp_neb"
        worker.run(structures)
        worker.inspect(structures)
        if worker.get_number_of_running_jobs() == 0:
            results = worker.retrieve(include_retrieved=True)
        else:
            results = []

    mid_atoms = results[0][-1][1]
    final_energy = mid_atoms.get_potential_energy()
    print(f"{final_energy = }")

    assert np.allclose([final_energy], [-82.195963])


if __name__ == "__main__":
    ...
