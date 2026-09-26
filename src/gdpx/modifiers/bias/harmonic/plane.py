#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import copy
from typing import Optional, List

import numpy as np

from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.geometry import find_mic


def compute_harmonic_energy_and_forces(
    cell, positions, harmonic_position, kspring: float, selected_indices: List[int],
    pbc: bool=True
):
    """"""
    # - compute forces
    distances, vectors = [], []
    for i in selected_indices:
        pos_i = copy.deepcopy(positions[i])
        pos_i[2] = 0.
        vec, dis = find_mic(harmonic_position - pos_i, cell, pbc=pbc)
        vectors.append(vec)
        distances.append(dis)
    distances = np.array(distances)

    # - compute energy
    energy = np.sum(0.5*kspring*distances**2)

    # - compute forces
    forces = np.zeros(positions.shape)
    for i, vec in zip(selected_indices, vectors):
        frc_i = kspring*vec
        forces[i] += frc_i

    return energy, forces


class PlaneHarmonicCalculator(Calculator):

    implemented_properties = ["energy", "free_energy", "forces"]

    def __init__(self, harmonic_position, kspring: float = 0.1, *args, **kwargs):
        """"""
        super().__init__(*args, **kwargs)

        #: Spring constant, eV/Ang^2.
        self.kspring = kspring

        #: Harmonic position on the plane.
        self.harmonic_position = np.array(harmonic_position)
        assert self.harmonic_position[2] == 0.

        return

    def calculate(
        self,
        atoms: Optional[Atoms] = None,
        properties=["energy"],
        system_changes=["positions"],
    ):
        """"""
        super().calculate(atoms, properties, system_changes)

        target_indices = [45, 46, 47]

        energy, forces = compute_harmonic_energy_and_forces(
            atoms.cell, atoms.positions, self.harmonic_position, self.kspring, 
            selected_indices=target_indices, pbc=True
        )

        print(f"{energy =}")
        print(f"{forces[target_indices, :] =}")

        self.results["energy"] = energy
        self.results["free_energy"] = energy
        self.results["forces"] = forces

        return


if __name__ == "__main__":
    ...
