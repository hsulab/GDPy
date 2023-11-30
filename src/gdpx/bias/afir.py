#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import List
from itertools import product

import numpy as np
from jax import numpy as jnp
from jax import grad, jit, value_and_grad

from ase import Atoms
from ase.data import covalent_radii
from ase.calculators.calculator import Calculator


"""Some calculators of external forces.
"""

eps_afir = 1.0061 / 96.485

@jit
def force_function(
    positions, covalent_radii, pair_indices, 
    pair_shifts, # in cartesian, AA
    gamma=2.5, r0=3.8164, epsilon=eps_afir
):
    """AFIR function."""
    bias = 0.0
    alpha = gamma/((2**(-1/6)-(1+(1+gamma/epsilon)**0.5)**(-1/6))*r0)

    # inverse distance weights
    pair_positions = jnp.take(positions, pair_indices, axis=0)
    #print(pair_positions.shape)
    dvecs = pair_positions[0] - pair_positions[1] + pair_shifts
    #print(dvecs)
    distances = jnp.linalg.norm(dvecs, axis=1)
    #print(distances)

    pair_radii = jnp.take(covalent_radii, pair_indices, axis=0)
    #print(pair_radii)
    #print((pair_radii[0]+pair_radii[1]))

    weights = ((pair_radii[0]+pair_radii[1])/distances)**6
    #print(weights)

    bias = alpha * jnp.sum(weights*distances) / jnp.sum(weights)
    #print(bias)

    return bias

#dfn = grad(force_function, argnums=0)
compute_afir = value_and_grad(force_function, argnums=0)


class AFIRCalculator(Calculator):

    implemented_properties = ["energy", "free_energy", "forces"]

    default_parameters = dict(
        gamma = 2.5, # eV,
        r0 = 3.8164, # Ar-Ar LJ
        epsilon = eps_afir,
        groups = None
    )
    
    def __init__(self, restart=None, label=None, atoms=None, directory='.', **kwargs):
        """"""
        super().__init__(restart=restart, label=label, atoms=atoms, directory=directory, **kwargs)

        return
    
    def calculate(self, atoms=None, properties=["energy"], system_changes=["positions","numbers","cell"]):
        """"""
        super().calculate(atoms, properties, system_changes)

        # - get constants
        atomic_numbers = atoms.get_atomic_numbers()
        atomic_radii = np.array([covalent_radii[i] for i in atomic_numbers])

        # - find reactive groups
        #groups = []
        #for group_command in self.parameters["groups"]:
        #    curr_group = create_a_group(atoms, group_command)
        #    assert len(curr_group) > 0, f"No atoms in group {group_command}."
        #    groups.append(curr_group)
        frag_indices = self.parameters["groups"]
        #print("groups: ", groups)

        pair_indices = list(product(*frag_indices))
        pair_indices = np.array(pair_indices).transpose()
        pair_shifts = np.zeros((pair_indices.shape[1],3)) # TODO: ...

        ret = compute_afir(
            atoms.positions, atomic_radii, 
            pair_indices, pair_shifts, self.parameters["gamma"],
            self.parameters["r0"], self.parameters["epsilon"]
        )
        self.results["energy"] = np.asarray(ret[0])
        self.results["free_energy"] = self.results["energy"]
        self.results["forces"] = -np.array(ret[1]) # copy forces

        return


if __name__ == "__main__":
    ...
