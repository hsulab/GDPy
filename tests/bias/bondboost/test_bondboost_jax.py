#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import itertools

import numpy as np

import jax
import jax.numpy as jnp


def compute_distances(positions, pair_indices):
    """"""
    pos_i = positions.take(pair_indices[0], axis=0)
    pos_j = positions.take(pair_indices[1], axis=0)

    vecs = pos_i - pos_j
    distances = jnp.linalg.norm(vecs, axis=1)

    return distances


def compute_boost_energy(
    positions,
    ref_distances,
    bond_pairs,
    vmax: float = 0.5,
    smax: float = 0.5,
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
    num_bonds = bond_pairs.shape[0]

    # - compute bond strains
    pos_i = positions.take(bond_pairs[:, 0], axis=0)
    pos_j = positions.take(bond_pairs[:, 1], axis=0)
    vecs = pos_i - pos_j

    distances = jnp.linalg.norm(vecs, axis=1)

    bond_strains = (distances - ref_distances) / ref_distances

    max_index = jnp.argmax(bond_strains)

    # - compute energy
    # V_b, shape (num_bonds, )
    vboost = vmax / num_bonds * (1 - (bond_strains / smax) ** 2)

    max_strain_ratio = bond_strains[max_index] / smax
    u = 1 - max_strain_ratio**2
    v = 1 - curv**2 * max_strain_ratio**2

    env = u * u / v
    energy = jnp.sum(vboost * env)

    return energy


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
    print(f"{positions =}")

    # - compute bonds
    bond_pairs = [(0, 1), (0, 2)]
    bond_pairs = np.array(bond_pairs)
    pair_indices = bond_pairs.T

    distances = compute_distances(positions, pair_indices)
    print(f"{distances =}")

    # O-C 1.1, O-O 1.2
    ref_distances = [1.1, 1.2]
    ref_distances = np.array(ref_distances)

    bond_strains = (distances - ref_distances) / ref_distances
    print(f"{bond_strains =}")

    max_index = np.argmax(bond_strains)
    print(f"{max_index =}")

    # -
    # energy = compute_boost_energy(positions, ref_distances, bond_pairs)
    energy, gradients = jax.value_and_grad(compute_boost_energy, argnums=0)(
        positions, ref_distances, bond_pairs
    )
    forces = -1.*gradients
    print(f"{energy =}")
    # compute_boost_forces = jax.grad(compute_boost_energy, argnums=0)
    # forces = -1*compute_boost_forces(positions, ref_distances, bond_pairs)
    np.savetxt("./frc.dat", forces, fmt="%12.4f")
    ...
