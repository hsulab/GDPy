#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import pytest

from gdpx.execution.factory import create_worker, create_workers

@pytest.fixture
def cp2k_config():
    """"""
    params = dict(
        schema_version=3,
        potential = dict(
            provider = "cp2k",
            parameters = dict(
                command = "srun /mnt/scratch2/chemistry-apps/dkb01416/cp2k/developed/cp2k-9.1/exe/local/cp2k.psmp",
                template = "/mnt/scratch2/users/40247882/porous/inputs/PBE+D3_RKS.inp",
                basis_set = "DZVP-MOLOPT-SR-GTH",
                basis_set_file = "/mnt/scratch2/chemistry-apps/dkb01416/cp2k/developed/cp2k-9.1/data/BASIS_MOLOPT",
                pseudo_potential = "GTH-PBE",
                potential_file = "/mnt/scratch2/chemistry-apps/dkb01416/cp2k/developed/cp2k-9.1/data/GTH_POTENTIALS"
            ),
        ),
        executor = dict(
            provider = "cp2k",
            method = "spc",
            parameters = dict(ignore_convergence = True),
        )
    )

    return params

def test_empty(cp2k_config):
    """"""
    worker = create_worker(cp2k_config)
    print(worker)

    driver = worker.driver
    driver.directory = "./assets/empty_cand"

    print(driver.read_convergence())

    return

def test_broken(cp2k_config):
    """"""
    worker = create_worker(cp2k_config)
    print(worker)

    driver = worker.driver
    driver.directory = "/mnt/scratch2/users/40247882/porous/nqtrain/r0/_explore/mimescn/0008.run_cp2k/cand18"

    print(driver.read_convergence())

    return

def test_broken_by_abort(cp2k_config):
    """"""
    worker = create_worker(cp2k_config)
    print(worker)
    print(worker.driver.ignore_convergence)

    driver = worker.driver
    driver.directory = "/mnt/scratch2/users/40247882/porous/nqtrain/r0/_explore/mimescn/0008.run_cp2k/cand74"

    print("calc: ", driver.calc.read_convergence())
    print("driver: ", driver.read_convergence())

    return


if __name__ == "__main__":
    ...
