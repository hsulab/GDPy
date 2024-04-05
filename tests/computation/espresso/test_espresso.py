#!/usr/bin/env python3
# -*- coding: utf-8 -*

import copy

import pytest
import tempfile


from ase.io import read, write

from gdpx.worker.interface import ComputerVariable



@pytest.fixture
def espresso_spc_config():
    """"""
    params = dict(
        potter = dict(
            name = "espresso",
            params = dict(
                backend = "espresso",
                command = "mpirun -n 2 pw.x -in PREFIX.pwi > PREFIX.pwo",
                pp_path = "/mnt/scratch2/chemistry-apps/dkb01416/espresso/pseudo/oncv_upf",
                pp_name = "_ONCV_PBE-1.2.upf",
                #kpts = [1, 1, 1],
                kspacing = 0.04,
            )
        ),
        driver = dict(
            backend = "ase"
        )
    )

    return params

def test_spc(espresso_spc_config):
    """"""
    config = espresso_spc_config
    atoms = read("../assets/H2.xyz")

    config = copy.deepcopy(config)
    worker = ComputerVariable(**config).value[0]

    driver = worker.driver
    driver.directory = "./xxx" # tmpdir

    _ = driver.run(atoms, ...)

    return


if __name__ == "__main__":
    ...
