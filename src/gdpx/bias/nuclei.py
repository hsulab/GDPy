#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from typing import Callable, List

import numpy as np
from ase.calculators.calculator import Calculator
from ase.calculators.morse import MorsePotential
from ase.neighborlist import NeighborList, natural_cutoffs

from .utils import get_bond_information, get_equidis_dict


def compute_repulsion_energy_and_forces(
    positions,
    bond_pairs,
    bond_distances,
    bond_shifts,
    bond_equi_distances,
    epsilon,
    rho0,
):
    """"""
    # compute energy
    expf = np.exp(rho0 * (1.0 - bond_distances / bond_equi_distances))
    energy = np.sum(epsilon * expf * (expf - 2.0))

    # compute forces
    forces = np.zeros(positions.shape)

    dEdr = 2.0 * epsilon * (expf - 1.0) * expf * (-rho0 / bond_equi_distances)

    for ip, (i, j) in enumerate(bond_pairs):
        vec_mic = positions[i] - (positions[j] + bond_shifts[ip])
        frc_ij = -dEdr[ip] * vec_mic / bond_distances[ip]
        forces[i, :] += frc_ij
        forces[j, :] += -frc_ij

    return energy, forces


class NucleiRepulsionCalculator(Calculator):

    implemented_properties = ["energy", "free_energy", "forces"]

    default_parameters = dict()

    def __init__(
        self,
        bonds,
        covalent_ratio=[0.8, 2.0],
        epsilon: float = 1.0,
        rho0: float = 6.0,
        check_atoms_change: bool = False,
        *args,
        **kwargs,
    ):
        """"""
        super().__init__(
            *args,
            **kwargs,
        )

        self.epsilon = epsilon
        self.rho0 = rho0

        self.cov_min, self.cov_max = covalent_ratio

        self.symbols, self.bonds, self.equidis_dict = get_equidis_dict(
            bonds, ratio=self.cov_min
        )

        self.check_atoms_change = check_atoms_change

        self.neighlist = None

        self.target_indices = None

        return

    def calculate(
        self, atoms=None, properties=["energy"], system_changes=["positions"]
    ):
        """"""
        super().calculate(atoms, properties, system_changes)

        # find target atoms
        if self.target_indices is None:
            self.target_indices = [
                i for i, a in enumerate(atoms) if a.symbol in self.symbols
            ]
        else:
            ...

        # create a neighlist
        if self.neighlist is None:
            self.neighlist = NeighborList(
                self.cov_min * np.array(natural_cutoffs(atoms)),
                skin=0.0,
                self_interaction=False,
                bothways=False,
            )
        else:
            ...

        # For some simulations, e.g. GCMC, the number of atoms are changing!!
        # We should manully set check on!!
        if self.check_atoms_change:
            num_atoms = len(atoms)
            num_stored_atoms = self.neighlist.nl.cutoffs.shape[0]
            if num_atoms != num_stored_atoms:
                self.neighlist = NeighborList(
                    self.cov_min * np.array(natural_cutoffs(atoms)),
                    skin=0.0,
                    self_interaction=False,
                    bothways=False,
                )
            self.target_indices = [
                i for i, a in enumerate(atoms) if a.symbol in self.symbols
            ]
        else:
            ...

        self.neighlist.update(atoms)

        bond_pairs, bond_distances, bond_shifts, bond_equi_distances = (
            get_bond_information(
                atoms,
                self.neighlist,
                self.equidis_dict,
                # covalent_min=self.cov_min,
                covalent_min=0.0,
                target_indices=self.target_indices,
                allowed_bonds=self.bonds,
            )
        )

        energy, forces = compute_repulsion_energy_and_forces(
            atoms.positions,
            bond_pairs,
            bond_distances,
            bond_shifts,
            bond_equi_distances,
            epsilon=self.epsilon,
            rho0=self.rho0,
        )

        self.results["energy"] = energy
        self.results["free_energy"] = energy
        self.results["forces"] = forces

        return


if __name__ == "__main__":
    ...
