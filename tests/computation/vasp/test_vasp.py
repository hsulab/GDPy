#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import List
import pathlib

import pytest
import tempfile
import yaml

from ase import Atoms
from ase.io import read, write

from gdpx.worker.interface import run_worker
from gdpx.worker.interface import ComputerVariable


@pytest.fixture
def create_vasp_config():

    def potter(command="mpirun -n 4 vasp_gam 2>&1 > vasp.out", driver={}):
        """"""
        pot_params = dict(
            name = "vasp",
            params = dict(
                backend = "vasp",
                command = command,
                incar = "/scratch/gpfs/jx1279/incars/INCAR_LABEL_NoMAG",
                kpts = 25,
                pp_path = "/home/jx1279/apps/vasp/potpaw/recommend",
                vdw_path = "/home/jx1279/apps/vasp/potpaw"
            )
        )

        params = {}
        params["potential"] = pot_params
        params["driver"] = driver

        return params
    
    return potter

@pytest.fixture
def water_molecule() -> List[Atoms]:
    """"""
    #atoms = molecule("H2O", vacuum=10.)

    atoms = Atoms(
        numbers = [8,1,1],
        positions = [
            [10.00000000, 10.76320000, 10.59630000],
            [10.00000000, 11.52650000, 10.00000000],
            [10.00000000, 10.00000000, 10.00000000],
        ],
        cell = [[19.,0.,0.], [0.,20.,0.], [0.,0.,21.]],
        pbc = False
    )

    return [atoms]

@pytest.fixture
def structures() -> List[Atoms]:
    """H2"""

    atoms = Atoms(
        numbers = [1,1],
        positions = [
            [10.00000000, 10.00000000, 10.72000000],
            [10.00000000, 10.00000000, 10.00000000],
        ],
        cell = [[9.,0.,0.], [0.,10.,0.], [0.,0.,11.]],
        pbc = True
    )

    return [atoms]

# @pytest.mark.basic
# @pytest.mark.vasp
def test_spc_driver(create_vasp_config, structures):
    """"""
    vasp_spc_params = create_vasp_config()
    with tempfile.NamedTemporaryFile(suffix=".xyz") as strtmp:
        write(strtmp.name, structures)

        with tempfile.NamedTemporaryFile(suffix=".yaml") as dptmp:
            with open(dptmp.name, "w") as fopen:
                yaml.safe_dump(vasp_spc_params, fopen)
            
            #worker = create_potter(config_file=dptmp.name)
            worker = None
            
        # - 
        with tempfile.TemporaryDirectory() as tmpdirname:
            #tmpdirname = "./asexxx"
            run_worker(strtmp.name, directory=tmpdirname, worker=worker)

            stored_atoms = read(pathlib.Path(tmpdirname)/"results"/"end_frames.xyz", ":")[0]
            stored_energy = stored_atoms.get_potential_energy()
    
    assert stored_energy == pytest.approx(-31.601453)


@pytest.mark.vasp
def test_vasp_md(structures):
    """"""
    with open("./assets/vaspmd.yaml", "r") as fopen:
        vasp_params = yaml.safe_load(fopen)

    # print(vasp_params)
    worker = ComputerVariable(**vasp_params).value[0]
    print(worker)

    print(structures)

    # - run
    with tempfile.TemporaryDirectory() as tmpdirname:
        # worker.directory = tmpdirname
        worker.directory = "./temp"
        worker.run(structures)
        worker.inspect(structures)
        if worker.get_number_of_running_jobs() == 0:
            results = worker.retrieve(include_retrieved=True)
        else:
            results = []

    print(results[0][0].get_potential_energy())

    return


if __name__ == "__main__":
    ...
