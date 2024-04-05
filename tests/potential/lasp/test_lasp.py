#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import copy
import pytest
import tempfile

from ase.io import read, write

from GDPy.core.register import import_all_modules_for_register
from GDPy.worker.interface import WorkerVariable


import_all_modules_for_register()


@pytest.fixture
def emt_config():
    """"""
    params = dict(
        potential = dict(
            name = "emt",
            #params = dict(
            #    backend = "emt"
            #)
        ),
        driver = dict(
            backend = "ase"
        )
    )

    return params

@pytest.fixture
def emt_md_config():
    """"""
    params = dict(
        potential = dict(
            name = "emt",
            #params = dict(
            #    backend = "emt"
            #)
        ),
        driver = dict(
            backend = "ase",
            ignore_convergence = True,
            task = "md",
            init = dict(
                velocity_seed = 1112,
                dump_period = 1
            ),
            run = dict(
                steps = 10
            )
        )
    )

    return params


def test_empty(emt_config):
    """"""
    worker = WorkerVariable(emt_config["potential"], emt_config["driver"]).value

    driver = worker.driver
    driver.directory = "./assets/empty_driver"

    converged = driver.read_convergence()

    assert not converged

def test_broken_spc(emt_config):
    """"""
    worker = WorkerVariable(emt_config["potential"], emt_config["driver"]).value

    driver = worker.driver
    driver.directory = "./assets/broken_ase_spc"

    #print("broken: ", driver.read_convergence())

    #atoms = read("../assets/Cu-fcc-s111p22.xyz")
    #new_atoms = driver.run(atoms, read_exists=True)
    #print(driver.read_convergence())

    converged = driver.read_convergence()

    assert not converged

def test_finished_spc(emt_config):
    """"""
    worker = WorkerVariable(emt_config["potential"], emt_config["driver"]).value

    driver = worker.driver
    driver.directory = "./assets/finished_ase_spc"

    converged = driver.read_convergence()

    #atoms = read("../assets/Cu-fcc-s111p22.xyz")
    #new_atoms = driver.run(atoms, read_exists=True)
    #print(driver.read_convergence())

    assert converged


def test_finished_md(emt_md_config):
    """"""
    with tempfile.TemporaryDirectory() as tmpdir:
        # - run 10 steps
        config = copy.deepcopy(emt_md_config)
        worker = WorkerVariable(config["potential"], config["driver"]).value

        driver = worker.driver
        driver.directory = tmpdir

        converged = driver.read_convergence()
        print("before: ", converged)

        atoms = read("../assets/Cu-fcc-s111p22.xyz")

        new_atoms = driver.run(atoms, read_exists=True)
        print(driver.read_convergence())

    return

def test_restart_md(emt_md_config):
    """"""
    # - run 10 steps
    with tempfile.TemporaryDirectory() as tmpdir:
        #tmpdir = "./xxx"
        config = copy.deepcopy(emt_md_config)
        worker = WorkerVariable(config["potential"], config["driver"]).value

        driver = worker.driver
        driver.directory = tmpdir

        converged = driver.read_convergence()
        print("before: ", converged)

        atoms = read("../assets/Cu-fcc-s111p22.xyz")

        new_atoms = driver.run(atoms, read_exists=True)
        print("10 steps: ", driver.read_convergence())

        # - run extra 10 steps within the same dir
        config = copy.deepcopy(emt_md_config)
        config["driver"]["run"]["steps"] = 20
        print("new: ", config)

        worker = WorkerVariable(config["potential"], config["driver"]).value

        driver = worker.driver
        driver.directory = tmpdir

        converged = driver.read_convergence()
        print("restart: ", converged)

        atoms = read("../assets/Cu-fcc-s111p22.xyz")

        new_atoms = driver.run(atoms, read_exists=True)
        print(driver.read_convergence())

        traj = driver.read_trajectory()
        print(traj[5].get_kinetic_energy())

        assert len(traj) == 21


if __name__ == "__main__":
    ...