#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import copy
import pytest
import pathlib
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
def nequip_lammps_config():
    """"""
    params = dict(
        potential = dict(
            name = "nequip",
            params = dict(
                backend = "lammps",
                commad = "lmp -in in.lammps 2>&1 -in in.lammps",
                type_list = ["C", "H", "N", "O", "S"],
                model = ["/mnt/scratch2/users/40247882/porous/nqtrain/r0/_ensemble/0004.train/m0/nequip.pth"]
            )
        ),
        driver = dict(
            ignore_convergence = True,
            task = "md",
            init = dict(
                velocity_seed = 1112,
            ),
            run = dict(
                steps = 10
            )
        )
    )

    return params

@pytest.fixture
def deepmd_lammps_config():
    """"""
    params = dict(
        potential = dict(
            name = "deepmd",
            params = dict(
                backend = "lammps",
                command = "lmp -in in.lammps 2>&1 > lmp.out",
                type_list = ["Al", "Cu", "O"],
                model = [
                    "../assets/dpmd-AlCuO-m0.pb",
                    "../assets/dpmd-AlCuO-m1.pb",
                ]
            )
        ),
        driver = dict(
            ignore_convergence = True,
            task = "md",
            init = dict(
                velocity_seed = 1112,
                dump_period = 2
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


def test_finished_md(deepmd_lammps_config):
    """"""
    config = deepmd_lammps_config
    with tempfile.TemporaryDirectory() as tmpdir:
        # - run 10 steps
        config = copy.deepcopy(config)
        worker = WorkerVariable(config["potential"], config["driver"]).value

        driver = worker.driver
        driver.directory = tmpdir

        converged = driver.read_convergence()
        #print("before: ", converged)

        atoms = read("../assets/Cu-fcc-s111p22.xyz")

        new_atoms = driver.run(atoms, read_exists=True)
        #print(driver.read_convergence())

        assert pytest.approx(-55.988444) == new_atoms.get_potential_energy()

def test_restart_md(deepmd_lammps_config):
    """"""
    config = deepmd_lammps_config
    #atoms = read("../assets/methanol.xyz")
    atoms = read("../assets/Cu-fcc-s111p22.xyz")
    # - run 10 steps
    with tempfile.TemporaryDirectory() as tmpdir:
        #tmpdir = "./xxx"

        config = copy.deepcopy(config)
        worker = WorkerVariable(config["potential"], config["driver"]).value

        driver = worker.driver
        driver.directory = tmpdir

        converged = driver.read_convergence()
        print("before: ", converged)

        new_atoms = driver.run(atoms, read_exists=True)
        print("10 steps: ", driver.read_convergence())
        traj = driver.read_trajectory()
        print("nframes: ", len(traj))
        print(new_atoms.get_potential_energy())

        # - run extra 10 steps within the same dir
        config = copy.deepcopy(config)
        config["driver"]["run"]["steps"] = 20
        print("new: ", config)

        worker = WorkerVariable(config["potential"], config["driver"]).value

        driver = worker.driver
        driver.directory = tmpdir

        converged = driver.read_convergence()
        print("restart: ", converged)

        new_atoms = driver.run(atoms, read_exists=True)
        print(driver.read_convergence())
        print("config: ", driver.setting)

        traj = driver.read_trajectory()
        #print(traj[10].get_kinetic_energy())
        
        write(pathlib.Path(tmpdir)/"out.xyz", traj)

        assert pytest.approx(1.117415e-02) == traj[-1].info["max_devi_f"]

        


if __name__ == "__main__":
    ...