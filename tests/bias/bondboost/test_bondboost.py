#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import itertools

import numpy as np


def compute_distances(positions, pair_indices):
    """"""
    pos_i = positions.take(pair_indices[0], axis=0)
    pos_j = positions.take(pair_indices[1], axis=0)

    vecs = pos_i - pos_j
    distances = np.linalg.norm(vecs, axis=1)

    return distances


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
    print(f"{vboost =}")

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


if __name__ == "__main__":
    """"""
    # - init system
    positions = [
        [8.46464846, 7.90128836, 9.74705502],  # O
        [8.52906984, 7.93583778, 8.57023828],  # C
        [4.45833644, 6.31306244, 7.93158728],  # O
        [4.54489395, 7.15497821, 7.09078878],  # C
    ]
    positions = np.array(positions).reshape(-1, 3)
    # print(f"{positions =}")

    # - compute bonds
    bond_pairs = [(0, 1), (0, 2)]
    pair_indices = np.array(bond_pairs).T

    distances = compute_distances(positions, pair_indices)
    print(f"{distances =}")

    # O-C 1.1, O-O 1.2
    ref_distances = [1.1, 1.2]

    bond_strains = (distances - ref_distances) / ref_distances
    print(f"{bond_strains =}")

    max_index = np.argmax(bond_strains)
    print(f"{max_index =}")

    # -
    energy, forces = compute_boost_energy_and_forces(
        positions, distances, ref_distances, bond_pairs, bond_strains, max_index
    )
    np.savetxt("./frc.dat", forces, fmt="%12.4f")
    print(f"{energy =}")
    ...
