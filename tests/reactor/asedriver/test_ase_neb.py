#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import copy
import tempfile

import pytest
import yaml

import numpy as np

from ase.build import add_adsorbate, fcc100
from ase.calculators.emt import EMT
from ase.constraints import FixAtoms
from ase.optimize import QuasiNewton

from gdpx.cli.compute import convert_config_to_potter


@pytest.mark.basic
def test_vasp_neb():
    """"""
    # - prepare structures
    ini = fcc100("Al", size=(2, 2, 3))
    add_adsorbate(ini, "Au", 1.7, "hollow")
    ini.center(axis=2, vacuum=4.0)

    # FIXME: Check whether IS and FS are the same...
    fin = copy.deepcopy(ini)
    fin[-1].x += fin.get_cell()[0, 0] / 2.0

    # FIXME: Better pathway input structures...
    structures = [ini, fin]

    for atoms in structures:
        atoms.set_constraint(FixAtoms(indices=range(8)))
        atoms.calc = EMT()
        qn = QuasiNewton(atoms, trajectory=None)
        qn.run(fmax=0.08)

    # -
    with open("./assets/aseneb.yaml", "r") as fopen:
        emt_params = yaml.safe_load(fopen)

    worker = convert_config_to_potter(emt_params)
    print(f"{worker =}")

    with tempfile.TemporaryDirectory() as tmpdirname:
        worker.directory = tmpdirname
        # worker.directory = "./test_ase_neb"
        worker.run(structures)
        worker.inspect(structures)
        if worker.get_number_of_running_jobs() == 0:
            results = worker.retrieve(include_retrieved=True)
        else:
            results = []

    mid_atoms = results[0][-1][1]
    final_energy = mid_atoms.get_potential_energy()
    print(f"{final_energy = }")

    assert np.allclose([final_energy], [3.698085])

    ...
