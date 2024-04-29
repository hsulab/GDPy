#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import copy
from typing import List, Optional, Tuple

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.geometry import find_mic

from .. import str2array
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
    dx = dis - center
    energy = 0.5 * kspring * dx**2

    # compute forces
    forces = np.zeros((2, 3))
    frc_i = -kspring * dx * vec / dis
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
        assert isinstance(self.center, float)

        self.kspring = kspring
        assert isinstance(self.kspring, float)

        return

    def _icalculate(self, atoms, properties, system_changes) -> Tuple[dict, list]:
        """"""
        vec, dis = compute_distance(atoms.cell, atoms.positions[self.group], pbc=True)

        energy = 0.0
        forces = np.zeros(atoms.positions.shape)
        if self.num_steps >= self.delay:
            energy, ext_forces = compute_distance_harmonic_energy_and_forces(
                vec, dis, self.center, self.kspring
            )
            forces = np.zeros(atoms.positions.shape)
            forces[self.group] = ext_forces
        else:
            ...

        results = {}
        results["energy"] = energy
        results["free_energy"] = energy
        results["forces"] = forces

        step_info = (self.num_steps, dis, energy)

        return results, step_info

    def _write_first_step(self):
        """"""
        content = f"# {self.pace =} {self.group =} {self.center =} {self.kspring =}\n"
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

    @staticmethod
    def broadcast_params(inp_dict: dict) -> List[dict]:
        """"""
        # broadcast center or kspring
        centers = inp_dict.get("center", [])
        if isinstance(centers, float):
            centers = [centers]
        elif isinstance(centers, str):
            centers = str2array(centers)
        else:
            raise TypeError(f"{centers =}")
        num_centers = len(centers)

        ksprings = inp_dict.get("kspring", [])
        if isinstance(ksprings, float):
            ksprings = [ksprings]
        elif isinstance(ksprings, str):
            ksprings = str2array(ksprings)
        else:
            raise TypeError(f"{ksprings =}")
        num_ksprings = len(ksprings)

        new_inputs = []
        if num_centers == 1 and num_ksprings == 1:
            new_inputs = [inp_dict]
        elif num_centers == 1 and num_ksprings > 1:
            new_inputs = [copy.deepcopy(inp_dict) for _ in range(num_ksprings)]
            for i, x in enumerate(new_inputs):
                x["kspring"] = ksprings[i]
        elif num_centers > 1 and num_ksprings == 1:
            new_inputs = [copy.deepcopy(inp_dict) for _ in range(num_centers)]
            for i, x in enumerate(new_inputs):
                x["center"] = centers[i]
        else:
            raise RuntimeError("Broadcast cannot.")

        return new_inputs

    @staticmethod
    def broadcast(inp_dict: dict) -> List["DistanceHarmonicCalculator"]:
        """Create a list of calculators based on input parameters."""

        new_inputs = DistanceHarmonicCalculator.broadcast_params(inp_dict)
        calcs = [DistanceHarmonicCalculator(**x) for x in new_inputs]

        return calcs


if __name__ == "__main__":
    ...
