#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from typing import Optional

import numpy as np

from ase import Atoms
from ase import units
from ase.constraints import FixAtoms
from ase.md.md import MolecularDynamics
from ase.md.verlet import VelocityVerlet
from ase.md.langevin import Langevin


class TimeStampedMonteCarlo(MolecularDynamics):

    def __init__(
        self,
        atoms: Atoms,
        maxstepsize: float = 0.20,
        fixcm: bool = True,  # TODO:
        temperature: Optional[float] = None,
        temperature_K: Optional[float] = None,
        rng=np.random.Generator(np.random.PCG64()),
        *args,
        **kwargs,
    ):
        """Time-stamped force-bias Monte Carlo.

        Args:
            atoms: Atoms object.

        """
        MolecularDynamics.__init__(self, atoms=atoms, *args, **kwargs)

        self.maxstepsize = maxstepsize

        self.fix_com = fixcm

        self.temp = units.kB * self._process_temperature(
            temperature, temperature_K, "eV"
        )  # == kbT

        self.rng = rng

        # - get minimum mass
        #   no more new atoms should be added after the simulation is defined.
        self.mass_min = np.min(self.masses)

        return

    def step(self, forces=None):
        """"""
        atoms = self.atoms

        # NOTE: The md forces does not apply FixAtoms, 
        #       so all forces are physical
        if forces is None:
            # forces = atoms.get_forces(apply_constraint=True, md=False)
            forces = atoms.get_forces(apply_constraint=True, md=True)
        # print(f"forces: {forces}")

        # - get displacement steps
        stepsizes = self.maxstepsize * np.power(self.mass_min / self.masses, 0.25)

        natoms = len(atoms)
        displacements = np.zeros((natoms, 3))

        for i in range(natoms):
            for j in range(3):
                P_acc, P_ran = 0, 1
                gamma = forces[i, j] * stepsizes[i] / (2.0 * self.temp)
                gamma_exp = np.exp(gamma)
                gamma_expi = 1.0 / gamma_exp
                while P_acc < P_ran:
                    xi = 2.0 * self.rng.uniform(-1.0, 1.0)
                    P_ran = self.rng.uniform()
                    if xi < 0:
                        P_acc = np.exp(2.0 * xi * gamma) * gamma_exp - gamma_expi
                        P_acc = P_acc / (gamma_exp - gamma_expi)
                    elif xi > 0:
                        P_acc = gamma_expi - np.exp(2.0 * xi * gamma) * gamma_expi
                        P_acc = P_acc / (gamma_exp - gamma_expi)
                    else:
                        P_acc = 1.0

                    displacements[i][j] = xi * stepsizes[i]

        positions = atoms.get_positions()

        # print(f"prev_com: {atoms.get_center_of_mass()}")

        # - remove translation
        if self.fix_com:
            for constraint in atoms.constraints:
                if isinstance(constraint, FixAtoms):
                    move_indices = [
                        i for i in range(natoms) if i not in constraint.index
                    ]
                    break
            else:
                move_indices = list(range(natoms))
            com_disp = (
                self.masses[move_indices].T
                @ displacements[move_indices, :]
                / np.sum(self.masses[move_indices])
            )
            displacements[move_indices, :] -= com_disp
            # prev_com = atoms.get_center_of_mass()

        # - remove rotation
        ...

        # - adjust posistions
        atoms.set_positions(positions + displacements)

        # print(f"move_com: {atoms.get_center_of_mass()}")

        # NOTE: ase v3.22.1 has a bug in set_center_of_mass
        #       so we do it by ourselves
        # NOTE: set_com should consider constraints!!!
        # if self.fix_com:
        #     diff_com = prev_com - atoms.get_center_of_mass()
        #     print(f"diff_com: {diff_com}")
        #     print(f"move_com: {atoms.get_center_of_mass()}")
        #     atoms.set_positions(
        #         atoms.get_positions() + diff_com, apply_constraint=False
        #     )

        # print(f"curr_com: {atoms.get_center_of_mass()}")

        return forces


if __name__ == "__main__":
    ...
