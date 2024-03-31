#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import List

import numpy as np

from ase import Atoms
from ase import data, units
from ase.neighborlist import NeighborList, natural_cutoffs

from .operator import AbstractOperator


class MoveOperator(AbstractOperator):

    name: str = "move"

    def __init__(
        self,
        particles: List[str] = None,
        region: dict = {},
        temperature: float = 300.0,
        pressure: float = 1.0,
        covalent_ratio=[0.8, 2.0],
        max_disp: float = 2.0,
        use_rotation: bool = True,
        *args,
        **kwargs,
    ):
        """"""
        super().__init__(
            region=region,
            temperature=temperature,
            pressure=pressure,
            covalent_ratio=covalent_ratio,
            use_rotation=use_rotation,
            *args,
            **kwargs,
        )

        self.particles = particles

        self.max_disp = max_disp

        return

    def run(self, atoms: Atoms, rng=np.random) -> Atoms:
        """"""
        super().run(atoms)
        self._extra_info = "-"

        # BUG: If there is no species in the system...
        species_indices = self._select_species(atoms, self.particles, rng=rng)

        # - basic
        curr_atoms = atoms.copy()  # TODO: use clean atoms?
        cell = curr_atoms.get_cell(complete=True)

        # - neighbour list
        nl = NeighborList(
            self.covalent_max * np.array(natural_cutoffs(curr_atoms)),
            skin=0.0,
            self_interaction=False,
            bothways=True,
        )

        # - find tag atoms
        # record original position of species_indices
        species = curr_atoms[species_indices]
        self._extra_info = f"Move_{species.get_chemical_formula()}_{species_indices}"

        # org_pos = new_atoms[species_indices].position.copy() # original position
        # TODO: deal with pbc, especially for move step
        org_com = np.mean(species.positions, axis=0)
        org_positions = species.positions.copy()

        # - move the atom
        for i in range(self.MAX_RANDOM_ATTEMPTS):
            rsq = 1.1
            while rsq > 1.0:
                rvec = 2 * rng.uniform(size=3) - 1.0
                rsq = np.linalg.norm(rvec)
            ran_pos = org_com + rvec * self.max_disp
            # -- make a copy and rotate
            species_ = self._rotate_species(species, rng=rng)
            curr_cop = np.average(species_.positions, axis=0)
            # -- translate
            new_vec = ran_pos - curr_cop
            species_.translate(new_vec)
            curr_atoms.positions[species_indices] = species_.positions.copy()
            # use neighbour list
            if not self.check_overlap_neighbour(nl, curr_atoms, cell, species_indices):
                self._print(f"succeed to random after {i+1} attempts...")
                self._print(f"original position: {org_com}")
                self._print(f"random position: {ran_pos}")
                self._print(
                    f"actual position: {np.average(curr_atoms.positions[species_indices], axis=0)}"
                )
                break
            curr_atoms.positions[species_indices] = org_positions
        else:
            curr_atoms = None

        return curr_atoms

    def metropolis(self, prev_ene: float, curr_ene: float, rng=np.random) -> bool:
        """"""
        # - acceptance ratio
        kBT_eV = units.kB * self.temperature
        beta = 1.0 / kBT_eV  # 1/(kb*T), eV

        coef = 1.0
        ene_diff = curr_ene - prev_ene
        acc_ratio = np.min([1.0, coef * np.exp(-beta * (ene_diff))])

        # content = "\nVolume %.4f Nexatoms %.4f CubicWave %.4f Coefficient %.4f\n" %(
        #    self.acc_volume, len(self.tag_list[expart]), cubic_wavelength, coef
        # )
        content = "\nVolume %.4f Beta %.4f Coefficient %.4f\n" % (
            self.region.get_volume(),
            beta,
            coef,
        )
        content += "Energy Difference %.4f [eV]\n" % ene_diff
        content += "Accept Ratio %.4f\n" % acc_ratio
        for x in content.split("\n"):
            self._print(x)

        rn_move = rng.uniform()
        self._print(f"{self.__class__.__name__} Probability %.4f" % rn_move)

        return rn_move < acc_ratio

    def as_dict(self) -> dict:
        """"""
        params = super().as_dict()
        params["particles"] = self.particles
        params["max_disp"] = self.max_disp

        return params

    def __repr__(self) -> str:
        """"""
        content = f"@Modifier {self.__class__.__name__}\n"
        content += (
            f"temperature {self.temperature} [K] pressure {self.pressure} [bar]\n"
        )
        content += "covalent ratio: \n"
        content += f"  min: {self.covalent_min} max: {self.covalent_max}\n"
        content += f"max disp: {self.max_disp}\n"
        content += f"particles: \n"
        content += f"  {self.particles}\n"

        return content


if __name__ == "__main__":
    ...
