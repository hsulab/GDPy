#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import List
from itertools import product

import numpy as np
from jax import numpy as jnp
from jax import grad, jit

from ase import Atoms
from ase.data import covalent_radii
from ase.calculators.calculator import Calculator

from GDPy.builder.group import create_a_group


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

dfn = grad(force_function, argnums=0)

class AFIRBias():

    def __init__(self, groups: List[str]=[], gamma=2.5, r0=3.8164, epsilon=eps_afir, *args, **kwargs):
        """"""
        self.atoms = None

        self.group_commands = groups # define what groups react
        assert len(self.group_commands) == 2, f"{self.__class__.__name__} needs two groups."

        self.gamma = gamma

        return
    
    def attach_atoms(self, atoms: Atoms):
        """Attach atoms for bias computation and update related parameters."""
        # - attach atoms
        self.atoms = atoms
        atomic_numbers = self.atoms.get_atomic_numbers()
        self.atomic_radii = np.array([covalent_radii[i] for i in atomic_numbers])

        # - find reactive groups
        groups = []
        for group_command in self.group_commands:
            groups.append(create_a_group(atoms, group_command))
        self._groups = groups

        return
    
    def compute(self):
        """"""
        assert self.atoms is not None, "Bias has no attached atoms to compute."

        frag_indices = self._groups

        pair_indices = list(product(*frag_indices))
        pair_indices = np.array(pair_indices).transpose()
        pair_shifts = np.zeros((pair_indices.shape[1],3))

        # TODO: check positions of molecules since 
        #       PBC may break them if a small cell is used...

        ext_forces = -dfn(
            self.atoms.positions, self.atomic_radii, 
            pair_indices, pair_shifts, self.gamma
        )

        return ext_forces


class AFIRCalculator(Calculator):

    implemented_properties = ["energy", "free_energy","forces"]

    default_parameters = dict(
        gamma = 2.5, # eV,
        r0 = 3.8164, # Ar-Ar LJ
        epsilon = eps_afir,
        groups = None
    )
    
    def __init__(self, restart=None, label=None, atoms=None, directory='.', **kwargs):
        """"""
        super().__init__(restart=restart, label=label, atoms=atoms, directory=directory, **kwargs)
        print("params: ", self.parameters)

        return
    
    def calculate(self, atoms=None, properties=["energy"], system_changes=["positions","numbers","cell"]):
        """"""
        super().calculate(atoms, properties, system_changes)

        # - get constants
        atomic_numbers = atoms.get_atomic_numbers()
        atomic_radii = np.array([covalent_radii[i] for i in atomic_numbers])

        # - find reactive groups
        groups = []
        for group_command in self.parameters["groups"]:
            curr_group = create_a_group(atoms, group_command)
            assert len(curr_group) > 0, f"No atoms in group {group_command}."
            groups.append(curr_group)
        frag_indices = groups
        #print("groups: ", groups)

        pair_indices = list(product(*frag_indices))
        pair_indices = np.array(pair_indices).transpose()
        pair_shifts = np.zeros((pair_indices.shape[1],3)) # TODO: ...

        ext_energy = force_function(
            atoms.positions, atomic_radii, 
            pair_indices, pair_shifts, self.parameters["gamma"],
            self.parameters["r0"], self.parameters["epsilon"]
        )
        ext_energy = float(ext_energy)
        self.results["energy"] = ext_energy
        self.results["free_energy"] = ext_energy
        #print("ext_energy: ", ext_energy)

        ext_forces = -dfn(
            atoms.positions, atomic_radii, 
            pair_indices, pair_shifts, self.parameters["gamma"],
            self.parameters["r0"], self.parameters["epsilon"]
        )
        self.results["forces"] = ext_forces
        #print("ext_forces", ext_forces)

        return


if __name__ == "__main__":
    ...