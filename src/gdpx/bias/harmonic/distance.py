#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import copy
from typing import Optional, Tuple, List

import numpy as np

from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.geometry import find_mic

from ..timeio import TimeIOCalculator


def compute_distance(cell, positions, pbc: bool = True):
    """"""
    # compute colvar
    assert positions.shape[0] == 2
    vec, dis = find_mic(positions[0] - positions[1], cell=cell)

    return vec, dis


def compute_distance_harmonic_energy_and_forces(
    vec, dis: float, center: float, kspring: float
):
    """"""
    # compute energy
    energy = np.sum(0.5 * kspring * (dis - center) ** 2)

    # compute forces
    forces = np.zeros((2, 3))
    frc_i = -kspring * vec / dis
    forces[0] += frc_i
    forces[1] += -frc_i

    return energy, forces


class DistanceHarmonicCalculator(TimeIOCalculator):

    implemented_properties = ["energy", "free_energy", "forces"]

    def __init__(
        self, group: List[int], center: float, kspring: float = 0.1, *args, **kwargs
    ):
        """"""
        super().__init__(*args, **kwargs)

        num_group_atoms = len(group)
        assert num_group_atoms == 2
        self.group = group

        self.center = center

        self.kspring = kspring

        return

    def _icalculate(self, atoms, properties, system_changes) -> Tuple[dict, list]:
        """"""
        vec, dis = compute_distance(atoms.cell, atoms.positions[self.group], pbc=True)

        energy, ext_forces = compute_distance_harmonic_energy_and_forces(
            vec, dis, self.center, self.kspring
        )
        forces = np.zeros(atoms.positions.shape)
        forces[self.group] = ext_forces

        results = {}
        results["energy"] = energy
        results["free_energy"] = energy
        results["forces"] = forces

        step_info = (self.num_steps, dis, energy)

        return results, step_info

    def _write_first_step(self):
        """"""
        content = "# {:>10s}  {:>12s}  {:>12s}\n".format("step", "distance", "energy")
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