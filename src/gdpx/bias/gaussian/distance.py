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
    dsdx1 = vec / dis
    dsdx = np.vstack([dsdx1, -dsdx1]).reshape(s.shape[0], -1, 3)

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

    v0 = omega * np.exp(-np.sum(s2, axis=1))[:, np.newaxis]
    v = np.sum(v0)

    dvds = np.sum(-v0 * s1 / sigma**2, axis=0)  # shape (num_dim, )

    return v, dvds


def compute_bias_forces(dvds, dsdx):
    """"""
    # dvds (num_dim, ) dsdx (num_dim, num_atoms, 3)
    forces = -np.sum(
        np.tile(dvds[:, np.newaxis, np.newaxis], dsdx.shape[1:]) * dsdx, axis=0
    )

    return forces


class DistanceGaussianCalculator(TimeIOCalculator):

    implemented_properties = ["energy", "free_energy", "forces"]

    def __init__(self, group: List[int], width: float, height: float, *args, **kwargs):
        """"""
        super().__init__(*args, **kwargs)

        num_group_atoms = len(group)
        assert num_group_atoms == 2
        self.group = group

        self.width = np.array(width)[np.newaxis]
        self.height = height

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
                s, np.array(self._history_records), self.width, self.height
            )
            forces[self.group] = compute_bias_forces(dvds, dsdx)
        else:
            ...

        results = {}
        results["energy"] = energy
        results["free_energy"] = energy
        results["forces"] = forces

        step_info = (self.num_steps, s[0], energy)

        return results, step_info

    def _write_first_step(self):
        """"""
        content = f"# {self.pace =} {self.group =} {self.width =} {self.height =}\n"
        content += "# {:>10s}  {:>12s}  {:>12s}\n".format("step", "distance", "energy")
        with open(self.log_fpath, "w") as fopen:
            fopen.write(content)

        return

    def _write_step(self):
        """"""
        content = "{:>12d}  {:>12.4f}  {:>12.4f}\n".format(*self.step_info)
        with open(self.log_fpath, "a") as fopen:
            fopen.write(content)

        return


if __name__ == "__main__":
    ...
