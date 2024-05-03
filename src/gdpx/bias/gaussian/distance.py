#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from typing import List, Tuple

import numpy as np
from ase.geometry import find_mic

from ..timeio import TimeIOCalculator


def compute_colvar_and_gradient(cell, positions, pbc: bool = True):
    """Compute colvar and gradient.

    Returns:
        s: (num_dim, )
        dsdx: (num_dim, num_atoms, 3)

    """
    # compute colvar
    assert positions.shape[0] == 2
    vec, dis = find_mic(positions[0] - positions[1], cell=cell)

    s = np.array(dis)[np.newaxis]
    dsdx = np.vstack([vec / dis, -vec / dis]).reshape(s.shape, -1, 3)

    return s, dsdx


def compute_gaussian_and_gradient(s, s_t, sigma, omega):
    """Computa gaussian and gradient.

    Args:
        s: Colvar (num_dim, ).
        s_t: (num_records, num_dim).
        sigma: (num_dim, )
        omega: scalar.

    """
    s1 = s - s_t
    s2 = s1**2 / 2.0 / sigma**2
    v = np.sum(omega * np.exp(-np.sum(s2, axis=1)))

    dvds = np.sum(-v * s1 / sigma**2, axis=0)  # shape (num_dim, )

    return v, dvds


class DistanceGaussianCalculator(TimeIOCalculator):

    implemented_properties = ["energy", "free_energy", "forces"]

    def __init__(self, group: List[int], sigma: float, omega: float, *args, **kwargs):
        """"""
        super().__init__(*args, **kwargs)

        num_group_atoms = len(group)
        assert num_group_atoms == 2
        self.group = group

        self.sigma = np.array(sigma)[np.newaxis]
        self.omega = omega

        self._history_records = []

        return

    def _icalculate(self, atoms, properties, system_changes) -> Tuple[dict, tuple]:
        """"""
        s, dsdx = compute_colvar_and_gradient(
            atoms.cell, atoms.positions[self.group], pbc=True
        )

        energy = 0.0
        forces = np.zeros(atoms.positions.shape)
        if self.num_steps >= self.delay:
            if self.num_steps % self.pace == 0:
                self._history_records.append(s)
            energy, dvds = compute_gaussian_and_gradient(
                s, self._history_records, self.sigma, self.omega
            )
            # dvds (num_dim, ) dsdx (num_dim, num_atoms, 3)
            forces[self.group] = -np.sum(
                np.tile(dvds[:, np.newaxis, np.newaxis], dsdx.shape[1:]) * dvds, axis=0
            )
        else:
            ...

        results = {}
        results["energy"] = energy
        results["free_energy"] = energy
        results["forces"] = forces

        step_info = (self.num_steps, s, energy)

        return results, step_info


if __name__ == "__main__":
    ...
