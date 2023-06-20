#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import copy
import pytest
import pathlib
import tempfile

from ase.io import read, write

from GDPy.core.register import import_all_modules_for_register
from GDPy.worker.interface import ComputerVariable


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
def vasp_md_config():
    """"""
    params = dict(
        potential = dict(
            name = "vasp",
            params = dict(
                backend = "vasp",
                command = "mpirun -n 4 /mnt/scratch2/chemistry-apps/dkb01416/vasp/installed/intel-2016/5.4.4-ccc/vasp_gam 2>&1 > vasp.out",
                incar = "./INCAR",
                kpts = [1,1,1],
                pp_path = "/mnt/scratch2/chemistry-apps/dkb01416/vasp/PseudoPotential",
                vdw_path = "/mnt/scratch2/chemistry-apps/dkb01416/vasp/PseudoPotential"
            )
        ),
        driver = dict(
            ignore_convergence = True,
            task = "md",
            init = dict(
                md_style = "nvt",
                temp = 300,
                velocity_seed = 1112,
                dump_period = 1
            ),
            run = dict(
                steps = 5
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
    #new_atoms = driver.run(atoms, read_exists=True)
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
    #new_atoms = driver.run(atoms, read_exists=True)
    #print(driver.read_convergence())

    assert converged


def test_finished_md(emt_md_config):
    """"""
    with tempfile.TemporaryDirectory() as tmpdir:
        # - run 10 steps
        config = copy.deepcopy(emt_md_config)
        worker = ComputerVariable(config["potential"], config["driver"]).value[0]

        driver = worker.driver
        driver.directory = tmpdir

        converged = driver.read_convergence()
        print("before: ", converged)

        atoms = read("../assets/Cu-fcc-s111p22.xyz")

        new_atoms = driver.run(atoms, read_exists=True)
        print(driver.read_convergence())

    return

def test_restart_md(vasp_md_config):
    """"""
    config = vasp_md_config
    #atoms = read("../assets/methanol.xyz")
    atoms = read("../assets/H2.xyz")
    # - run 10 steps
    with tempfile.TemporaryDirectory() as tmpdir:
        # tmpdir = "./xxx"

        config = copy.deepcopy(config)
        worker = ComputerVariable(config["potential"], config["driver"]).value[0]

        driver = worker.driver
        driver.directory = tmpdir

        converged = driver.read_convergence()
        print("before: ", converged)

        new_atoms = driver.run(atoms, read_exists=True)
        print("10 steps: ", driver.read_convergence())
        traj = driver.read_trajectory()
        print("nframes: ", len(traj))
        print("temp: ", traj[-1].get_temperature())

        # - run extra 10 steps within the same dir
        config = copy.deepcopy(config)
        config["driver"]["run"]["steps"] = 10
        print("new: ", config)

        worker = ComputerVariable(config["potential"], config["driver"]).value[0]

        driver = worker.driver
        driver.directory = tmpdir

        converged = driver.read_convergence()
        print("restart: ", converged)

        new_atoms = driver.run(atoms, read_exists=True)
        print(driver.read_convergence())

        traj = driver.read_trajectory()
        #print(traj[10].get_kinetic_energy())
        
        write(pathlib.Path(tmpdir)/"out.xyz", traj)

        assert pytest.approx(-6.6439426) == traj[-1].get_potential_energy()


if __name__ == "__main__":
    ...