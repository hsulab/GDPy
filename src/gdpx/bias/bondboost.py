#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import copy

from typing import Optional

import numpy as np

from ase import Atoms
from ase.calculators.calculator import Calculator


def compute_bond_matrix(atoms: Atoms):
    """"""
    distance_matrix = atoms.get_all_distances(mic=False, vector=False)
    distance_matrix = copy.deepcopy(distance_matrix)

    # self-strain will be None, avoid true-divide in strain
    np.fill_diagonal(distance_matrix, -1.0)

    return


def get_bond_information(atoms: Atoms):
    """"""
    new_bond_matrix = compute_bond_matrix(atoms)
    bond_strain = (new_bond_matrix - self.ref_bond_matrix) / self.ref_bond_matrix

    num_pairs = bond_strain.shape[0]  # == natoms?

    # - find the pair with the max bond strain
    max_index, max_bond_strain = [0, 0], 1e-8

    # check bond strain is smaller than limit, and bond distance is close enough,
    # and bonds should be formed by different reactants
    valid_pairs = []
    for i in range(num_pairs):
        for j in range(i + 1, num_pairs):
            if (
                np.fabs(bond_strain[i, j]) < self.bond_strain_limit
                and new_bond_matrix[i, j] < 3.0
            ):
                valid_pairs.append((i, j))
                if np.fabs(bond_strain[i, j]) > np.fabs(max_bond_strain):
                    max_bond_strain = bond_strain[i, j]
                    max_index = [i, j]

    return


def compute_boost_energy_and_forces(
    positions,
    distances,
    ref_distances,
    bond_pairs,
    bond_strains,
    max_index: int,
    vmax=0.5,
    smax=0.5,
    curv: float = 0.98,
):
    """Compute bond-boost energy.

    u  = (1-(eps_max/smax)^2)
    dU = -2*(eps_max/smax^2)
    v  = (1-curv^2*(eps_max/smax)^2)
    dV = -2*curv^2*eps_max/smax^2

    The envelope function:
        A = u * u / v

    Args:
        ...

    Examples:
        >>>

    """
    num_bonds = len(bond_pairs)
    num_atoms = positions.shape[0]

    # - compute energy
    # V_b, shape (num_bonds, )
    vboost = vmax / num_bonds * (1 - (bond_strains / smax) ** 2)

    max_strain_ratio = bond_strains[max_index] / smax
    u = 1 - max_strain_ratio**2
    v = 1 - curv**2 * max_strain_ratio**2

    env = u * u / v
    energy = np.sum(vboost * env)

    # - compute forces
    forces = np.zeros((num_atoms, 3))

    # dV_b/deps_i, shape (num_bonds, )
    d_vboost = -vmax / num_bonds * 2 * bond_strains / smax**2

    # -- shared terms
    for p, (i, j) in enumerate(bond_pairs):
        frc_ij = (
            -env
            * d_vboost[p]
            * (positions[i] - positions[j])
            / distances[p]
            / ref_distances[p]
        )
        forces[i] += frc_ij
        forces[j] += -frc_ij

    # -- the extra term for max_index
    du = -2 * max_strain_ratio / smax
    dv = -2 * curv**2 * max_strain_ratio / smax

    denv = du * (u / v) + u * (du * v - u * dv) / v**2

    max_i, max_j = bond_pairs[max_index]
    max_frc_ij = (
        -denv
        * (positions[max_i] - positions[max_j])
        / distances[max_index]
        / ref_distances[max_index]
        * np.sum(vboost)
    )
    forces[max_i] += max_frc_ij
    forces[max_j] += -max_frc_ij

    return energy, forces


class BondBoostCalculator(Calculator):

    implemented_properties = ["energy", "free_energy", "forces"]

    def __init__(
        self,
        vmax: float = 0.5,
        smax: float = 0.5,
        curv: float = 0.98,
        *args,
        **kwargs,
    ):
        """"""
        super().__init__(*args, **kwargs)

        self.ref_bond_matrix = None

        # bond-boost params
        #:V_max, eV
        self.vmax = vmax

        #: q, maximum bond change compared to the reference state
        self.smax = smax

        #: control the curvature near the boundary
        self.curv = curv

        return

    def calculate(
        self, atoms=None, properties=["energy"], system_changes=["positions"]
    ):
        """"""
        super().calculate(atoms, properties, system_changes)

        # - find bonds to boost
        bond_pairs = [[48, 49], [48, 50]]
        distances = []
        for (i, j) in bond_pairs:
            dis = atoms.get_distance(i, j, mic=True)
            distances.append(dis)
        distances = np.array(distances)
        ref_distances = np.array([1.1, 1.2])

        bond_strains = (distances - ref_distances) / ref_distances
        max_index = np.argmax(bond_strains)

        # print(f"{distances =}")
        # print(f"{bond_strains =}")
        # print(f"{max_index}")

        # - compute properties
        num_bonds = len(bond_pairs)
        if num_bonds > 0:
            energy, forces = compute_boost_energy_and_forces(
                atoms.positions,
                distances,
                ref_distances,
                bond_pairs,
                bond_strains,
                max_index,
                vmax=self.vmax,
                smax=self.smax,
                curv=self.curv,
            )
            # print(f"{energy =}")
            # print(f"{forces[[48, 49, 50, 51], :] =}")
        else:
            energy = 0.
            forces = np.zeros((atoms.positions.shape))

        self.results["energy"] = energy
        self.results["free_energy"] = energy
        self.results["forces"] = forces

        return


if __name__ == "__main__":
    ...
