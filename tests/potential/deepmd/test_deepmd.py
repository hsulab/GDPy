#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tempfile
import pathlib

import pytest
import yaml

from ase import Atoms
from ase.io import read, write
from ase.build import molecule

from gdpx.cli.compute import run_worker, ComputerVariable
from gdpx.utils.command import parse_input_file


DRIVER_PARAMS = dict(
    task = "min",
    run = dict(
        fmax = 0.05,
        steps = 400
    )
)


@pytest.fixture
def create_pot_config():

    def potter(backend, command, driver={}):
        """"""
        pot_params = dict(
            name = "deepmd",
            params = dict(
                backend = backend,
                command = command,
                type_list = ["Al", "Cu", "O"],
                model = [
                    "../assets/AlCuO-deepmd-r4.pb"
                ]
            )
        )

        params = {}
        params["potential"] = pot_params
        params["driver"] = driver

        return params
    
    return potter


@pytest.fixture
def structures() -> dict:
    """"""
    atoms = Atoms(
        numbers = [29]*13,
        positions = [
            [ 8.25644004,       8.24521998,      9.95633994],
            [ 8.23872000,       9.95654990,      8.24458002], 
            [ 8.16990009,      11.85999991,      9.99444003], 
            [ 8.24530002,       9.96308006,     11.90019004],  
            [ 9.96807003,       8.29306995,      8.23140005], 
            [11.85062995,       8.25667997,     10.15484999],  
            [10.09979994,       8.15879990,     11.86603993], 
            [11.84586000,       9.96839002,      8.20367008], 
            [10.06813004,      11.86674003,      8.34782998], 
            [11.81260005,      11.80219995,      9.99524008],
            [11.89553995,       9.96446999,     11.83908991], 
            [10.10144992,      11.83862010,     11.94890000], 
            [10.02733997,      10.03512008,     10.14950001]
        ],
        cell = [[19.,0.,0.], [0.,20.,0.], [0.,0.,21.]],
        #pbc = False
        pbc = True
    )

    return [atoms]


@pytest.mark.parametrize("backend,command", [["ase",""],["lammps","lmp -in in.lammps 2>&1 > lmp.out"]])
def test_spc_driver(create_pot_config, backend, command, structures):
    """"""
    dpmd_spc_params = create_pot_config(backend, command)
    with tempfile.NamedTemporaryFile(suffix=".xyz") as strtmp:
        write(strtmp.name, structures)

        with tempfile.NamedTemporaryFile(suffix=".yaml") as dptmp:
            with open(dptmp.name, "w") as fopen:
                yaml.safe_dump(dpmd_spc_params, fopen)
            
            params = parse_input_file(input_fpath=dptmp.name)
            worker = ComputerVariable(
                params["potential"], params.get("driver", {}), params.get("scheduler", {}),
                params.get("batchsize", 1)
            ).value[0]
            
        # - 
        with tempfile.TemporaryDirectory() as tmpdirname:
            #tmpdirname = "./asexxx"
            run_worker(strtmp.name, directory=tmpdirname, worker=worker)

            stored_atoms = read(pathlib.Path(tmpdirname)/"results"/"end_frames.xyz", ":")[0]
            stored_energy = stored_atoms.get_potential_energy()
    
    assert stored_energy == pytest.approx(-31.601453)


@pytest.mark.parametrize(
    "backend,command,driver", 
    [
        ["ase", "", DRIVER_PARAMS], 
        ["lammps", "lmp -in in.lammps 2>&1 > lmp.out", DRIVER_PARAMS]
    ]
)
def test_min_driver(create_pot_config, backend, command, driver, structures):
    """"""
    print(backend,command,driver)
    dpmd_spc_params = create_pot_config(backend, command, driver)
    with tempfile.NamedTemporaryFile(suffix=".xyz") as strtmp:
        write(strtmp.name, structures)

        with tempfile.NamedTemporaryFile(suffix=".yaml") as dptmp:
            with open(dptmp.name, "w") as fopen:
                yaml.safe_dump(dpmd_spc_params, fopen)
            
            params = parse_input_file(input_fpath=dptmp.name)
            worker = ComputerVariable(
                params["potential"], params.get("driver", {}), params.get("scheduler", {}),
                params.get("batchsize", 1)
            ).value[0]
            
        # - 
        with tempfile.TemporaryDirectory() as tmpdirname:
            #tmpdirname = "./asexxx"
            run_worker(strtmp.name, directory=tmpdirname, worker=worker)

            stored_atoms = read(pathlib.Path(tmpdirname)/"results"/"end_frames.xyz", ":")[0]
            stored_energy = stored_atoms.get_potential_energy()
    
    assert stored_energy == pytest.approx(-32.20, abs=1e-2)


if __name__ == "__main__":
    ...
