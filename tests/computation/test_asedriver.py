#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import copy
import logging
import pytest
import tempfile

from ase.io import read, write

from GDPy import config
from GDPy.core.register import import_all_modules_for_register
from GDPy.worker.interface import ComputerVariable


import_all_modules_for_register()
config.logger.setLevel(logging.DEBUG)


@pytest.fixture
def emt_config():
    """"""
    params = dict(
        potential = dict(
            name = "emt",
            params = dict(
                backend = "ase"
            )
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
            params = dict(
                backend = "ase"
            )
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

@pytest.fixture
def emt_min_config():
    """"""
    params = dict(
        potential = dict(
            name = "emt",
            params = dict(
                backend = "ase"
            )
        ),
        driver = dict(
            backend = "ase",
            ignore_convergence = True,
            task = "min",
            run = dict(
                fmax = 0.10,
                steps = 400
            )
        )
    )

    return params


def test_empty(emt_config):
    """"""
    worker = ComputerVariable(emt_config["potential"], emt_config["driver"]).value[0]

    driver = worker.driver
    driver.directory = "./assets/empty_driver"

    converged = driver.read_convergence()

    assert not converged

def test_broken_spc(emt_config):
    """"""
    worker = ComputerVariable(emt_config["potential"], emt_config["driver"]).value[0]

    driver = worker.driver
    driver.directory = "./assets/broken_ase_spc"

    #print("broken: ", driver.read_convergence())

    #atoms = read("../assets/Cu-fcc-s111p22.xyz")
    #new_atoms = driver.run(atoms, read_ckpt=True)
    #print(driver.read_convergence())

    converged = driver.read_convergence()

    assert not converged

def test_finished_spc(emt_config):
    """"""
    worker = ComputerVariable(emt_config["potential"], emt_config["driver"]).value[0]

    driver = worker.driver
    driver.directory = "./assets/finished_ase_spc"

    converged = driver.read_convergence()

    #atoms = read("../assets/Cu-fcc-s111p22.xyz")
    #new_atoms = driver.run(atoms, read_ckpt=True)
    #print(driver.read_convergence())

    assert converged


def test_finished_md(emt_md_config):
    """"""
    with tempfile.TemporaryDirectory() as tmpdir:
        #tmpdir = "./xxx"
        # - run 10 steps
        config = copy.deepcopy(emt_md_config)
        worker = ComputerVariable(config["potential"], config["driver"]).value[0]

        driver = worker.driver
        driver.directory = tmpdir

        converged = driver.read_convergence()
        print("before: ", converged)

        atoms = read("../assets/Cu-fcc-s111p22.xyz")

        new_atoms = driver.run(atoms, read_ckpt=True)
        print(driver.read_convergence())

    return

def test_finished_min(emt_min_config):
    """"""
    config = emt_min_config
    with tempfile.TemporaryDirectory() as tmpdir:
        #tmpdir = "./xxx"
        # - run 10 steps
        config = copy.deepcopy(config)
        worker = ComputerVariable(config["potential"], config["driver"]).value[0]

        driver = worker.driver
        driver.directory = tmpdir

        converged = driver.read_convergence()
        print("before: ", converged)

        atoms = read("../assets/Cu-fcc-s111p22.xyz")

        new_atoms = driver.run(atoms, read_ckpt=True)
        print(driver.read_convergence())

    return


@pytest.mark.parametrize("dump_period,nframes", [(1, 21), (3, 7), (5, 5)])
def test_restart_md(emt_md_config, dump_period, nframes):
    """"""
    # - run 10 steps
    with tempfile.TemporaryDirectory() as tmpdir:
        #tmpdir = "./xxx"
        config = copy.deepcopy(emt_md_config)
        config["driver"]["init"]["dump_period"] = dump_period
        worker = ComputerVariable(config["potential"], config["driver"]).value[0]

        driver = worker.driver
        driver.directory = tmpdir

        converged = driver.read_convergence()
        print("before: ", converged)

        atoms = read("../assets/Cu-fcc-s111p22.xyz")

        new_atoms = driver.run(atoms, read_ckpt=True)
        print("10 steps: ", driver.read_convergence())

        # - run extra 10 steps within the same dir
        config = copy.deepcopy(emt_md_config)
        config["driver"]["init"]["dump_period"] = dump_period
        config["driver"]["run"]["steps"] = 20
        print("new: ", config)

        worker = ComputerVariable(config["potential"], config["driver"]).value[0]

        driver = worker.driver
        driver.directory = tmpdir

        converged = driver.read_convergence()
        print("restart: ", converged)

        atoms = read("../assets/Cu-fcc-s111p22.xyz")

        new_atoms = driver.run(atoms, read_ckpt=True)
        print(driver.read_convergence())

        traj = driver.read_trajectory()
        #print(traj[5].get_kinetic_energy())

        assert len(traj) == nframes


if __name__ == "__main__":
    ...
