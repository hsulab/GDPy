#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import List

import numpy as np

from ase import Atoms
from ase.data import covalent_radii
from ase.calculators.calculator import Calculator
from ase.neighborlist import NeighborList, natural_cutoffs

from ..utils import compute_distance_and_shift


""""""

#: Argon cluster LJ epsilon.
EPS_AFIR: float = 1.0061 / 96.485

#: Argon cluster LJ radius.
R0_AFIR: float = 3.8164  # Ang

#: Argon cluster LJ power.
POW_AFIR: int = 6


def compute_afir_energy_and_forces(
    positions,
    distances,
    covalent_radii,
    pair_indices,
    pair_shifts,
    *,
    gamma: float = 2.5,
    power: int = POW_AFIR,
    r0: float = R0_AFIR,
    epsilon: float = EPS_AFIR,
):
    """AFIR function in JCTC2011.

    E_afir = \alpha*\frac{\sum_ij ((c_i+c_j)/r_ij)^p*r_ij}{\sum_ij ((c_i+c_j)/r_ij)^p}

    Args:
        pair_indices: shape (num_pairs, 2)
        pair_shifts:  in cartesian coordinates.

    """
    # - compute the prefactor alpha
    alpha = gamma / (
        (2 ** (-1 / 6) - (1 + (1 + gamma / epsilon) ** 0.5) ** (-1 / 6)) * r0
    )

    # - compute pair distances
    # pair_positions = np.take(positions, pair_indices.T, axis=0)
    # dvecs = pair_positions[0] - pair_positions[1] + pair_shifts
    # distances = np.linalg.norm(dvecs, axis=1)

    # - compute inverse distances
    pair_radii = np.take(covalent_radii, pair_indices.T, axis=0)
    weights = ((pair_radii[0] + pair_radii[1]) / distances) ** power

    # - compute energy
    w_s = np.sum(weights)
    dw_s = np.sum(weights * distances)
    energy = alpha * dw_s / w_s

    # - compute forces
    #   w = ((c_i+c_j)/r_ij)^p
    #   dE/dr = ((w+dw*r)*sum_w-sum_(w*r)*dw)/sum_w^2
    #   dE/dr *= alpha
    forces = np.zeros(positions.shape)
    for k, (i, j) in enumerate(pair_indices):
        pos_i, pos_j = positions[i], positions[j] + pair_shifts[k]
        r = distances[k]
        w = weights[k]
        d_w = -power / r * w
        dEdr = alpha * ((w + r * d_w) * w_s - dw_s * d_w) / w_s**2
        frc_ij = -dEdr * (pos_i - pos_j) / r
        forces[i] += frc_ij
        forces[j] -= frc_ij

    return energy, forces


class AFIRCalculator(Calculator):

    implemented_properties = ["energy", "free_energy", "forces"]

    default_parameters = dict(gamma=2.5, power=6, cutoff=6.0)

    def __init__(self, groups, use_pbc: bool=False, restart=None, label=None, atoms=None, directory=".", **kwargs):
        """"""
        super().__init__(
            restart=restart, label=label, atoms=atoms, directory=directory, **kwargs
        )

        self.gamma = self.parameters["gamma"]  # eV
        assert self.gamma > 0.

        self.radius = self.parameters.get("radius", R0_AFIR)
        self.epsilon = self.parameters.get("epsilon", EPS_AFIR)
        self.power = int(self.parameters.get("power", POW_AFIR))

        self.groups = groups

        self.use_pbc = use_pbc
        self.cutoff = self.parameters["cutoff"]

        assert len(self.groups) == 2

        # ---
        self.neighlist = None

        return

    def calculate(
        self,
        atoms=None,
        properties=["energy"],
        system_changes=["positions", "numbers", "cell"],
    ):
        """"""
        super().calculate(atoms, properties, system_changes)

        # NOTE: check cutoff before assigning groups
        if self.use_pbc:
            if self.neighlist is None:
                self.neighlist = NeighborList(
                    [self.cutoff/2.]*len(atoms),
                    skin=0.0, self_interaction=False, bothways=True
                )
            self.neighlist.update(atoms)
        else:
            self.neighlist = None

        # - get constants
        atomic_numbers = atoms.get_atomic_numbers()
        atomic_radii = np.array([covalent_radii[i] for i in atomic_numbers])

        # - find bond pairs
        bond_pairs, bond_distances, bond_shifts = compute_distance_and_shift(
            atoms, self.groups[0], self.groups[1], neighlist=self.neighlist
        )

        # - compute properties
        energy, forces = compute_afir_energy_and_forces(
            atoms.positions,
            bond_distances,
            atomic_radii,
            bond_pairs,
            bond_shifts,
            gamma=self.gamma,
            power=self.power,
            r0=self.radius,
            epsilon=self.epsilon,
        )

        # print(f"{energy =}")
        # print(f"{forces =}")

        self.results["energy"] = energy
        self.results["free_energy"] = energy
        self.results["forces"] = forces

        return


if __name__ == "__main__":
    ...
