#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from typing import List

import numpy as np

#: Argon cluster LJ epsilon.
EPS_AFIR: float = 1.0061 / 96.485

#: Argon cluster LJ radius.
R0_AFIR: float = 3.8164  # Ang


def compute_afir_energy_and_forces(
    positions,
    covalent_radii,
    pair_indices,
    pair_shifts,
    gamma: float = 2.5,
    power: int = 6,
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
    pair_positions = np.take(positions, pair_indices.T, axis=0)
    dvecs = pair_positions[0] - pair_positions[1] + pair_shifts
    distances = np.linalg.norm(dvecs, axis=1)

    # - compute inverse distances
    pair_radii = np.take(covalent_radii, pair_indices.T, axis=0)
    weights = ((pair_radii[0] + pair_radii[1]) / distances) ** power
    print(f"{weights =}")
    print(f"{weights.dtype =}")

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
        dEdr = alpha * ((w + r*d_w) * w_s - dw_s * d_w) / w_s**2
        frc_ij = -dEdr * (pos_i - pos_j) / r
        forces[i] += frc_ij
        forces[j] -= frc_ij

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
    positions = np.array(positions, dtype=np.float32).reshape(-1, 3)
    print(f"{positions.dtype =}")

    pair_indices = [[0, 2], [0, 3], [1, 2], [1, 3]]
    pair_indices = np.array(pair_indices)

    pair_shifts = np.zeros((4, 3))

    covalent_radii = [0.66, 0.76, 0.66, 0.76]
    covalent_radii = np.array(covalent_radii)

    # - compute bonds
    energy, forces = compute_afir_energy_and_forces(
        positions, covalent_radii, pair_indices, pair_shifts, gamma=2.5
    )
    print(f"{energy =}")
    print(f"{forces =}")
    ...
