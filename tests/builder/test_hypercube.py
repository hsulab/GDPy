#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
import pytest

import numpy as np

from jax import numpy as jnp
from jax import jacfwd, jacrev

from ase import Atoms

from GDPy.builder.hypercube import (
    compute_bond_distances, compute_bond_angles, pseudo_inverse_of_jacobian
)
from GDPy.builder.hypercube import HypercubeBuilder


@pytest.fixture(scope="function")
def molecule():
    """Create a CO2 molecule."""
    atoms = Atoms(
        symbols="CO2", 
        positions=[[0.,0.,0.],[1.2,0.01,0.],[-1.2,0.,0.]],
        cell=np.eye(3)*20.,
        pbc=True
    )

    return atoms

class TestInternalCoordinates:

    def test_distance(self, molecule: Atoms):
        """"""
        pairs = np.array([[0,1],[0,2]])
        distances = compute_bond_distances(molecule.positions, pairs)

        assert np.allclose(distances, [1.2000416659,1.2], atol=1e-6)
    
    def test_distance_jacobian_inverse(self, molecule: Atoms):
        """"""
        positions = molecule.positions
        natoms = len(molecule)

        pairs = np.array([[0,1],[0,2]])
        dis_jac_ = jacrev(compute_bond_distances, argnums=0)(positions, pairs)
        dis_jac = dis_jac_.reshape(-1,natoms*3)

        jac_inv = pseudo_inverse_of_jacobian(dis_jac)

        results = np.array(
            [
                [-0.3333029 ,  0.3333377],
                [-0.00555477, -0.00277715],
                [ 0.        ,  0.],
                [ 0.6665725 ,  0.33325806],             
                [ 0.00555477,  0.00277715],             
                [ 0.        ,  0.        ],             
                [-0.33326963, -0.66659576],             
                [ 0.        ,  0.        ],             
                [ 0.        ,  0.        ]
            ]
       )

        assert np.allclose(jac_inv, results, atol=1e-6)
    
    def test_angle(self, molecule: Atoms):
        """"""
        trimers = np.array([[0,1,2]])
        angles = compute_bond_angles(molecule.positions, trimers)

        assert np.allclose(angles, [3.1332633], atol=1e-6)
    
    def test_angle_jacobian_inverse(self, molecule: Atoms):
        """"""
        positions = molecule.positions
        natoms = len(molecule)

        trimers = np.array([[0,1,2]])
        ang_jac_ = jacrev(compute_bond_angles, argnums=0)(positions, trimers)
        ang_jac = ang_jac_.reshape(-1,natoms*3)

        jac_inv = pseudo_inverse_of_jacobian(ang_jac)

        results = np.array(
            [
                [-1.6656540e-03],             
                [ 3.9947912e-01],             
                [ 0.0000000e+00],             
                [ 1.6620012e-03],             
                [-1.9973263e-01],             
                [ 0.0000000e+00],             
                [ 3.6527499e-06],             
                [-1.9974649e-01],             
                [ 0.0000000e+00]
            ]
        )

        assert np.allclose(jac_inv, results, atol=1e-6)
    
    def test_iterative_position_update(self, molecule: Atoms):
        """"""
        natoms = len(molecule)

        tol = 1e-4
        pairs = np.array([[0,1],[0,2]])
        trimers = np.array([[0,1,2]])
        targets = np.array([1.5, 1.3, 3.14/3.])
        positions = copy.deepcopy(molecule.positions)
        for i in range(10):
            distances = compute_bond_distances(positions, pairs)
            angles = compute_bond_angles(positions, trimers)
            internals = np.hstack((distances,angles))
            disp = targets - internals 
            if np.max(np.fabs(disp)) < tol:
                break
            dis_jac_ = jacrev(compute_bond_distances, argnums=0)(positions, pairs)
            dis_jac = dis_jac_.reshape(-1,natoms*3)
            ang_jac_ = jacrev(compute_bond_angles, argnums=0)(positions, trimers)
            ang_jac = ang_jac_.reshape(-1,natoms*3)
            jac = np.vstack((dis_jac,ang_jac))
            jac_inv = pseudo_inverse_of_jacobian(jac)
            positions = copy.deepcopy(positions) + jnp.reshape(jac_inv@disp, (-1,3))
        else:
            ...
        
        results = np.array(
            [
                [-0.09825885, -0.7998146 ,  0.        ],             
                [ 0.7534894 ,  0.43493205,  0.        ],             
                [-0.65523046,  0.37488258,  0.        ]
            ]
        )

        assert np.allclose(positions, results, atol=1e-6)


class TestHyperCubeBuilder:

    def test_build_molecules(self, molecule):
        """"""
        # - create builder
        builder = HypercubeBuilder(
            distances=[[0,1],[0,2]], disrange=[0.8, 1.6],
            angles = [[0,1,2]], angrange=[75./180.*np.pi, 165./180.*np.pi],
            random_seed=1112
        )
        frames = builder.run([molecule], size=5)

        positions = np.array(
            [
                [ 0.00358386, -0.20563598,  0.        ],
                [ 1.14606953,  0.11257479,  0.        ],
                [-1.14965343,  0.10306119,  0.        ],
            ]
        )
        #print(frames[0].positions)
        #from ase.io import read, write
        #write("./xxx.xyz", frames)

        assert np.allclose(frames[0].positions, positions, atol=1e-6)


if __name__ == "__main__":
    ...
