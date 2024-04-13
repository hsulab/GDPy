#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from typing import List

import numpy as np

from ase.calculators.calculator import Calculator
from ase.geometry import find_mic


def compute_rmsd_energy_and_forces(
    cell, coordinate, saved_coordinates, sigma: float, omega: float, pbc: bool = True
):
    """"""
    # - preprocess coordinates
    num_atoms = len(coordinate)
    num_records = len(saved_coordinates)

    vectors, rmsd2 = [], []  # rmsd square
    if pbc:
        for i in range(num_records):
            mic_vec, mic_dis = find_mic(coordinate - saved_coordinates[i], cell)
            vectors.append(mic_vec)
            rmsd2_i = np.sum(mic_vec**2) / num_atoms
            rmsd2.append(rmsd2_i)
    else:
        for i in range(num_records):
            mic_vec = coordinate - saved_coordinates[i]
            vectors.append(mic_vec)
            rmsd2_i = np.sum(mic_vec**2) / num_atoms
            rmsd2.append(rmsd2_i)
    vectors = np.array(vectors)
    rmsd = np.sqrt(np.array(rmsd2))[:, np.newaxis]  # shape (num_references, 1)

    # - compute energy
    s1 = rmsd
    s2 = s1**2 / 2.0 / sigma**2  # uniform sigma?
    v = omega * np.exp(-np.sum(s2, axis=1))

    energy = v.sum(axis=0)

    # - compute forces
    # -- dE/ds _ shape (num_references, 1)
    dEds = -v[:, np.newaxis] * s1 / sigma**2

    # -- ds/dc # cv gradient wrt cartesian coordinate
    dsdc = (
        (1.0 / (num_atoms * rmsd))[:, :, np.newaxis]
        .repeat(num_atoms, axis=1)
        .repeat(3, axis=2)
    ) * vectors  # shape (num_references, num_atoms, 3)

    forces = -np.sum(dEds[:, :, np.newaxis] * dsdc, axis=0)

    return energy, forces


class RMSDGaussian(Calculator):

    implemented_properties = ["energy", "fee_energy", "forces"]

    def __init__(
        self,
        group: List[int],
        pace: int = 1,
        sigma: float = 0.2,
        omega: float = 0.5,
        **kwargs,
    ):
        """"""
        super().__init__(**kwargs)

        self.group = group

        self.pace = pace

        self.sigma = sigma

        self.omega = omega

        # - private
        self._history_records = []

        self._num_steps = 0

        return

    @property
    def num_steps(self) -> int:
        """"""

        return self._num_steps

    def calculate(
        self,
        atoms=None,
        properties=["energy"],
        system_changes=["positions", "numbers", "cell"],
    ):
        super().calculate(atoms, properties, system_changes)

        coordinate = atoms.get_positions()[self.group]
        if self.num_steps % self.pace == 0:
            self._history_records.append(coordinate)

        # The same coordinate leads to zero division so we use [:-1] records.
        saved_coordinates = np.array(self._history_records[:-1])
        if saved_coordinates.shape[0] == 0:
            energy = 0.
            forces = np.zeros(atoms.positions.shape)
        else:
            energy, biased_forces = compute_rmsd_energy_and_forces(
                cell=atoms.get_cell(complete=True),
                coordinate=coordinate,
                saved_coordinates=saved_coordinates,
                sigma=self.sigma,
                omega=self.omega,
                pbc=True,  # FIXME: nopbc systems?
            )
            forces = np.zeros(atoms.positions.shape)
            forces[self.group] = biased_forces

        self.results["energy"] = energy
        self.results["free_energy"] = energy
        self.results["forces"] = forces

        return


if __name__ == "__main__":
    ...
