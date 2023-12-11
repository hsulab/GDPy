#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
from typing import List
import warnings

import numpy as np
from scipy.stats import qmc

import jax
import jax.numpy as jnp
from jax import grad, value_and_grad, jit, jacfwd, jacrev

from ase import Atoms
from ase.io import read, write

from .builder import StructureModifier

"""Sample small molecule.
"""

@jit
def bond_distance(positions, indices: List[int]):
    """"""
    a1, a2 = indices
    dis = jnp.linalg.norm(positions[a2]-positions[a1])

    return dis
grad_bond_distance = grad(bond_distance, argnums=0)

@jit
def compute_bond_distances(positions: np.array, pair_indices: np.array):
    """Compute distances.

    Args:
        positions: Positions with shape of (natoms,3).
        pair_distances: Indices of atoms for distances [[0,1],[0,2]].
    
    Examples:

        >>> pairs = np.array([[0,1]])
        >>> distances = compute_bond_distances(positions, pairs)
    
    Returns:
        An array of distances.

    """
    pair_positions = jnp.take(positions, pair_indices.T, axis=0)
    dvecs = pair_positions[0] - pair_positions[1] # TODO: + pair_shifts
    distances = jnp.linalg.norm(dvecs, axis=1)

    return distances

@jit
def bond_angle(positions, indices):
    """"""
    a1, a2, a3 = indices
    dvec1 = positions[a2] - positions[a1]
    dvec2 = positions[a3] - positions[a1]

    #angle = jnp.arccos(jnp.dot(dvec1,dvec2)/jnp.linalg.norm(dvec1)/jnp.linalg.norm(dvec2))
    angle = jnp.arctan2(jnp.linalg.norm(jnp.cross(dvec1,dvec2)), jnp.dot(dvec1,dvec2))

    return angle

@jit
def compute_bond_angles(positions, trimer_indices):
    """Compute angles.

    For very small/acute angles, results by arccos are inaccurate. arctan may be 
    more effective.

    """
    trimer_positions = jnp.take(positions, trimer_indices.T, axis=0)
    # TODO: shifts
    dvecs1 = trimer_positions[1] - trimer_positions[0]
    dnorms1 = jnp.linalg.norm(dvecs1, axis=1)
    dvecs2 = trimer_positions[2] - trimer_positions[0]
    dnorms2 = jnp.linalg.norm(dvecs2, axis=1)

    angles = jnp.arccos(jnp.sum(dvecs1*dvecs2, axis=1) / dnorms1 / dnorms2)

    return angles

@jit
def pseudo_inverse_of_jacobian(jac, eps=0.0001):
    """"""
    dim = jac.shape[0]
    jac_inv = jnp.transpose(jac)@jnp.linalg.inv(jac@jnp.transpose(jac)+eps*jnp.eye(dim))

    return jac_inv


class HypercubeBuilder(StructureModifier):

    """Use hypercube sampling to generate internal coordinates.

    The implementation is not exact the same as the references. Here, we only 
    use hypercube to generate structures. No symmetry constraints is applied.

    TODO: 
        * Support dihedrals.

    References:
        [1] J. Chem. Phys. 2017, 147, 161706.
        [2] J. Chem. Phys. 2018, 149, 174114.

    """

    #: Maximum number of updates of positions.
    MAX_ATTEMPTS_UPDATE: int = 100

    #: Tolerance of target internal coordinates.
    TOL_INTCOORD: float = 1e-4

    def __init__(
        self, distances, disrange: List[float], 
        angles=None, angrange: List[float]=None, 
        substrates=None, *args, **kwargs
    ):
        """"""
        super().__init__(substrates=substrates, *args, **kwargs)

        self.dimers = np.array(distances)

        if angles:
            self.trimers = np.array(angles)
            self._use_ang = True
        else:
            self.trimers = []
            self._use_ang = False

        self.ndof = len(self.dimers) + len(self.trimers)

        bounds_ = []
        if disrange:
            if isinstance(disrange[0], list):
                bounds_.extend(disrange)
            else: # two floats
                bounds_.extend([disrange]*len(self.dimers))
        if self._use_ang:
            if angrange:
                if isinstance(angrange[0], list):
                    bounds_.extend(angrange)
                else:
                    bounds_.extend([angrange]*len(self.trimers))
            else:
                ... # TODO: set default 15,165?
        self.bounds = np.array(bounds_).T

        assert self.bounds.shape[1] == self.ndof, "Inconsistent number of freedoms and ranges."

        return
    
    def run(self, substrates: List[Atoms]=None, size: int=1, *args, **kwargs) -> List[Atoms]:
        """Modify input structures to generate random hypercube structures.
        """
        super().run(substrates=substrates, *args, **kwargs)

        frames = []
        for substrate in self.substrates:
            curr_frames = self._irun(substrate=substrate, size=size, *args, **kwargs)
            frames.extend(curr_frames)

        return frames
    
    def _irun(self, substrate: Atoms, size=1, *args, **kwargs) -> List[Atoms]:
        """"""
        # NOTE: strength new in 1.8.0
        #sampler = qmc.LatinHypercube(d=ninternals, strength=2, seed=self.rng) 
        sampler = qmc.LatinHypercube(d=self.ndof, seed=self.rng)
        samples = sampler.random(n=size)

        l_bounds, u_bounds = self.bounds
        scaled_samples = qmc.scale(samples, l_bounds, u_bounds)

        # - generate structures
        frames = []
        for sample in scaled_samples:
            curr_atoms = copy.deepcopy(substrate)
            curr_atoms = self._approx_structure(curr_atoms, sample, self.dimers, self.trimers)
            frames.append(curr_atoms)

        return frames
    
    def _approx_structure(self, atoms: Atoms, targets, pairs, trimers=None):
        """"""
        natoms = len(atoms)

        # - compute pseudo inverse of the jacobian matrix
        positions = copy.deepcopy(atoms.positions)
        for i in range(self.MAX_ATTEMPTS_UPDATE):
            distances = compute_bond_distances(positions, pairs)
            if self._use_ang:
                angles = compute_bond_angles(positions, trimers)
            else:
                angles = []
            internals = np.hstack((distances,angles))
            self._debug(f"internals: {internals}")
            disp = targets - internals 
            if np.max(np.fabs(disp)) < self.TOL_INTCOORD:
                break
            dis_jac_ = jacrev(compute_bond_distances, argnums=0)(positions, pairs)
            dis_jac = dis_jac_.reshape(-1,natoms*3)
            if self._use_ang:
                ang_jac_ = jacrev(compute_bond_angles, argnums=0)(positions, trimers)
                ang_jac = ang_jac_.reshape(-1,natoms*3)
                jac = np.vstack((dis_jac,ang_jac))
            else:
                ang_jac = []
                jac = dis_jac
            jac_inv = pseudo_inverse_of_jacobian(jac)
            positions = copy.deepcopy(positions) + jnp.reshape(jac_inv@disp, (-1,3))
        else:
            #warnings.warn("Iterative approximation is not converged.", UserWarning)
            self._print("Iterative approximation is not converged.")
        
        # - update positions
        atoms.positions = positions
        
        return atoms


if __name__ == "__main__":
    ...
