#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import copy
import logging
import pytest
import pathlib
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


def test_finished_md(deepmd_lammps_config):
    """"""
    config = deepmd_lammps_config
    with tempfile.TemporaryDirectory() as tmpdir:
        # - run 10 steps
        config = copy.deepcopy(config)
        worker = ComputerVariable(config["potential"], config["driver"]).value[0]

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
    dpconfig = deepmd_lammps_config
    dpconfig["driver"]["init"]["ckpt_period"] = 3
    atoms = read("../assets/Cu-fcc-s111p22.xyz")

    # - run 10 steps
    with tempfile.TemporaryDirectory() as tmpdir:
        #tmpdir = "./xxx"

        dpconfig = copy.deepcopy(dpconfig)
        worker = ComputerVariable(dpconfig["potential"], dpconfig["driver"]).value[0]

        driver = worker.driver
        driver.directory = tmpdir

        converged = driver.read_convergence()
        config._print(f"before: {converged}")

        new_atoms = driver.run(atoms, read_exists=True)
        config._print(f"10 steps: {driver.read_convergence()}")
        traj = driver.read_trajectory()
        config._print(f"nframes: {len(traj)}")
        config._print(new_atoms.get_potential_energy())

        # - run extra 10 steps within the same dir
        dpconfig = copy.deepcopy(dpconfig)
        dpconfig["driver"]["run"]["steps"] = 20
        config._print(f"new: {dpconfig}")

        worker = ComputerVariable(dpconfig["potential"], dpconfig["driver"]).value[0]

        driver = worker.driver
        driver.directory = tmpdir

        converged = driver.read_convergence()
        config._print(f"restart: {converged}")

        new_atoms = driver.run(atoms, read_exists=True)
        config._print(driver.read_convergence())
        config._print(f"config: {driver.setting}")

        traj = driver.read_trajectory()
        config._debug(f"nframes::: {len(traj)}")
        
        write(pathlib.Path(tmpdir)/"out.xyz", traj)

        assert pytest.approx(1.117415e-02) == traj[-1].info["max_devi_f"]

    
@pytest.mark.parametrize("dump_period,ckpt_period", [(2, 2), (2, 3), (4, 7)])
#@pytest.mark.parametrize("dump_period,ckpt_period", [(4, 7)])
def test_restart_md_steps(deepmd_lammps_config, dump_period, ckpt_period):
    """"""
    dpconfig = deepmd_lammps_config
    dpconfig["driver"]["init"]["ckpt_period"] = ckpt_period
    dpconfig["driver"]["init"]["dump_period"] = dump_period

    atoms = read("../assets/Cu-fcc-s111p22.xyz")

    # - run 10 steps
    with tempfile.TemporaryDirectory() as tmpdir:
        #tmpdir = "./xxx"

        dpconfig = copy.deepcopy(dpconfig)
        worker = ComputerVariable(dpconfig["potential"], dpconfig["driver"]).value[0]

        driver = worker.driver
        driver.directory = tmpdir

        converged = driver.read_convergence()
        config._print(f"before: {converged}")

        new_atoms = driver.run(atoms, read_exists=True)
        config._print(f"10 steps: {driver.read_convergence()}")
        traj = driver.read_trajectory()
        config._print(f"nframes: {len(traj)}")

        # - run extra 10 steps within the same dir
        dpconfig = copy.deepcopy(dpconfig)
        dpconfig["driver"]["run"]["steps"] = 20
        config._print(f"new: {dpconfig}")

        worker = ComputerVariable(dpconfig["potential"], dpconfig["driver"]).value[0]

        driver = worker.driver
        driver.directory = tmpdir

        converged = driver.read_convergence()
        config._print(f"restart: {converged}")

        new_atoms = driver.run(atoms, read_exists=True)
        config._print(driver.read_convergence())
        config._print(f"config: {driver.setting}")

        traj = driver.read_trajectory()
        config._debug(f"nframes::: {len(traj)}")
        
        write(pathlib.Path(tmpdir)/"out.xyz", traj)

        assert pytest.approx(1.117415e-02) == traj[-1].info["max_devi_f"]


if __name__ == "__main__":
    ...