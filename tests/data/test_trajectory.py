#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest
import tempfile

import numpy as np

from ase import Atoms
from ase.io import read, write
from ase.calculators.singlepoint import SinglePointCalculator

from GDPy.data.trajectory import Trajectory, Trajectories

@pytest.fixture
def md_config():
    """"""

    return dict(task="md", temp=300, timestep=10.)

@pytest.fixture
def create_images():
    def factory(energy=-2213.13439, forces=[[2131.,-123,4324.,],[12,4773,-2312.]]):
        """"""
        atoms = Atoms(
            numbers=[6,8], positions=[[0.,0.,0.],[0.,0.,1.]], cell=[[19.,0.,0.],[0.,20.,0.],[0.,0.,21.]],
            pbc=[0,0,1]
        )
        atoms.info["confid"] = 12313

        results = dict(energy=energy, forces=forces)
        spc = SinglePointCalculator(atoms, **results)
        atoms.calc = spc

        return [atoms]
    
    return factory

@pytest.mark.parametrize("energy", [-1.,2.,7.])
def test_save_and_load_trajectory(create_images, energy, md_config):
    """"""
    images = create_images(energy)
    trajectory = Trajectory(images, md_config)

    with tempfile.NamedTemporaryFile() as tmp:
        trajectory.save(tmp.name)
        rebuilt = Trajectory.from_file(tmp.name)

    rebuilt_atoms = rebuilt[0]

    assert images[0].get_potential_energy() == rebuilt_atoms.get_potential_energy()


if __name__ == "__main__":
    ...
