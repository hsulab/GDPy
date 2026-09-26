#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import copy
import dataclasses
import pathlib

from typing import List

import numpy as np

from ase.calculators.calculator import Calculator
from ase.geometry import find_mic


def compute_center_of_mass(
    cell, masses, positions, saved_positions, scaled: bool = False, pbc: bool = True
):
    """Compute center_of_mass in the fractional space.

    The positions should be properly processed with find_mic.

    Args:
        positions: The cartesian coordinates of a group of atoms.

    """

    shift = positions - saved_positions
    curr_vectors, curr_distances = find_mic(shift, cell, pbc=True)

    shifted_positions = positions + curr_vectors

    # dcom/dx = masses/np.sum(masses)
    com = masses @ shifted_positions / np.sum(masses)

    if scaled:
        com = cell.scaled_positions(com)
        for i in range(3):  # FIXME: seleced pbc?
            com[i] %= 1.0
            com[i] %= 1.0  # need twice see ase test

    return com


def compute_com_energy_and_forces(
    cell, masses, com, saved_coms, sigma: float, omega: float
):
    """"""
    # - compute energy
    x, x_t = com, saved_coms
    x1 = x - x_t
    x2 = x1**2 / 2.0 / sigma**2  # uniform sigma?
    v = omega * np.exp(-np.sum(x2, axis=1))

    energy = v.sum(axis=0)

    # - compute forces
    # -- dE/dx
    dEdx = np.sum(-v[:, np.newaxis] * x1 / sigma**2, axis=0)[np.newaxis, :]

    m = (masses / np.sum(masses))[:, np.newaxis]

    c_inv = np.linalg.inv(cell)
    forces = -np.transpose(c_inv @ (m @ dEdx).T)

    return energy, forces


class CenterOfMassGaussianCalculator(Calculator):

    implemented_properties = ["energy", "free_energy", "forces"]

    def __init__(
        self,
        groups: List[List[int]],
        sigma: List[float] = [0.05, 0.05, 1e6],
        omega: float = 0.2,
        scaled: bool = True,
        pace: int = 1,
        **kwargs,
    ):
        """Init center of mass gaussian."""
        super().__init__(**kwargs)

        self.groups = groups

        self.sigma = np.array(sigma)
        self.omega = omega

        self.scaled = scaled

        self.pace = pace

        # - private
        self._num_steps = 0

        self._saved_positions = None

        self._history_records = [[] for _ in range(len(self.groups))]

        return

    @property
    def num_steps(self) -> int:
        """"""

        return self._num_steps

    def reset_metadata(self):
        """Reset some simulation-related private attributes."""
        self._num_steps = 0
        self._saved_positions = None
        self._history_records = [[] for _ in range(len(self.groups))]

        return

    def calculate(
        self,
        atoms=None,
        properties=["energy"],
        system_changes=["positions", "numbers", "cell"],
    ):
        """"""
        super().calculate(atoms, properties, system_changes)

        # - save the intial group atoms position
        #   treat them as a whole molecule
        #   positions should be processed before using this calculator
        if self._saved_positions is None:
            self._saved_positions = []
            positions = copy.deepcopy(atoms.get_positions())
            for g in self.groups:
                self._saved_positions.append([positions[i] for i in g])

            content = ("# {:>10s}" + "{:>12s}  "*4 + "{:<s} \n").format(
                "step", "bias", "com_0", "com_1", "com_2", "tags"
            )

            log_fpath = pathlib.Path(self.directory) / "info.log"
            with open(log_fpath, "w") as fopen:
                fopen.write(content)

        # print(f"{self.num_steps =}")
        # - compute center_of_mass
        cell = atoms.get_cell(complete=True)
        masses = atoms.get_masses()
        positions = atoms.get_positions()
        for i, g in enumerate(self.groups):
            com = compute_center_of_mass(
                atoms.cell,
                masses[g],
                positions[g],
                saved_positions=self._saved_positions[i],
                scaled=self.scaled,
                pbc=True,
            )
            if self.num_steps % self.pace == 0:
                self._history_records[i].append(com)

        # - compute energy and forces
        energy = 0.
        forces = np.zeros(atoms.positions.shape)

        step_energies = []
        for i, g in enumerate(self.groups):
            saved_coms = np.array(self._history_records[i])
            com = saved_coms[-1]
            curr_energy, curr_forces = compute_com_energy_and_forces(
                cell, masses[g], com, saved_coms, sigma=self.sigma, omega=self.omega
            )
            energy += curr_energy
            step_energies.append(energy)
            for k, f in zip(g, curr_forces):
                forces[k] += f
        self._write_step(step_energies)

        # -
        self.results["energy"] = energy
        self.results["free_energy"] = energy
        self.results["forces"] = forces

        # - increase steps
        self._num_steps += 1

        return

    def _write_step(self, step_energies: List[float]):
        """"""
        content = ""
        for i, (g, ene, records) in enumerate(
            zip(self.groups, step_energies, self._history_records)
        ):
            content += ("{:>12d}  {:>12.4f}  " + "{:>12.4f} " * 3 + " {:<s}" + "\n").format(
                self._num_steps, ene, *records[-1], "_".join([str(x) for x in g])
            )

        log_fpath = pathlib.Path(self.directory) / "info.log"
        with open(log_fpath, "a") as fopen:
            fopen.write(content)

        return


if __name__ == "__main__":
    ...
