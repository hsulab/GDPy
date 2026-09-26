#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import pathlib
from typing import List

import numpy as np

from ase.calculators.calculator import Calculator
from ase.geometry import find_mic


def compute_rmsd_energy_and_forces(
    cell,
    coordinate,
    saved_coordinates,
    damped_steps: List[int],
    kappa: float,
    sigma: float,
    omega: float,
    pbc: bool = True,
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

    damped_prefactors = compute_damping_function(damped_steps, kappa=kappa)[
        :, np.newaxis, np.newaxis
    ]

    forces = -np.sum(damped_prefactors * dEds[:, :, np.newaxis] * dsdc, axis=0)

    return energy, forces


def compute_damping_function(steps, kappa: float = 0.03):
    """"""
    damp = 2.0 / (1.0 + np.exp(-kappa * steps)) - 1.0

    return damp


class RMSDGaussian(Calculator):

    implemented_properties = ["energy", "fee_energy", "forces"]

    def __init__(
        self,
        group: List[int],
        pace: int = 1,
        sigma: float = 0.2,
        omega: float = 0.5,
        kappa: float = 0.03,
        memory_length: int = 0,
        **kwargs,
    ):
        """"""
        super().__init__(**kwargs)

        self.group = group

        self.pace = pace

        self.sigma = sigma

        self.omega = omega

        self.kappa = kappa

        self.memory_length = memory_length

        # - private
        self._history_records = []
        self._history_steps = []

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

        if self.num_steps == 0:
            content = f"# pace {self.pace} width {self.sigma} height {self.omega}\n"
            content += f"# memory {self.memory_length} kappa {self.kappa}\n"
            content += "# {:>10s}  {:>12s}  {:>12s}\n".format(
                "step", "num_records", "energy"
            )

            log_fpath = pathlib.Path(self.directory) / "info.log"
            with open(log_fpath, "w") as fopen:
                fopen.write(content)

        coordinate = atoms.get_positions()[self.group]

        # Remove previous references
        self._history_records = self._history_records[-self.memory_length :]
        self._history_steps = self._history_steps[-self.memory_length :]

        # The same coordinate leads to zero division so we use [:-1] records
        # so we add record after computing energy and forces.
        saved_coordinates = np.array(self._history_records)
        damped_steps = self.num_steps - np.array(self._history_steps)
        if saved_coordinates.shape[0] == 0:
            energy = 0.0
            forces = np.zeros(atoms.positions.shape)
        else:
            energy, biased_forces = compute_rmsd_energy_and_forces(
                cell=atoms.get_cell(complete=True),
                coordinate=coordinate,
                saved_coordinates=saved_coordinates,
                damped_steps=damped_steps,
                kappa=self.kappa,
                sigma=self.sigma,
                omega=self.omega,
                pbc=True,  # FIXME: nopbc systems?
            )
            forces = np.zeros(atoms.positions.shape)
            forces[self.group] = biased_forces

        self.results["energy"] = energy
        self.results["free_energy"] = energy
        self.results["forces"] = forces

        self._write_step(energy)

        if self.num_steps % self.pace == 0:
            self._history_records.append(coordinate)
            self._history_steps.append(self.num_steps)

        self._num_steps += 1

        return

    def _write_step(self, energy: float):
        """"""
        num_records = len(self._history_records)
        content = "{:>12d}  {:>12d}  {:>12.4f}\n".format(
            self.num_steps, num_records, energy
        )

        log_fpath = pathlib.Path(self.directory) / "info.log"
        with open(log_fpath, "a") as fopen:
            fopen.write(content)

        return


if __name__ == "__main__":
    ...
