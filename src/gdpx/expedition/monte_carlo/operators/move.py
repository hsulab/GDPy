#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import copy
from typing import Optional

import numpy as np
from ase import Atoms, units
from ase.neighborlist import NeighborList, natural_cutoffs

from gdpx.geometry.bounce import get_a_random_direction
from gdpx.geometry.particle import translate_then_rotate
from gdpx.geometry.spatial import check_atomic_distances_by_neighbour_list

from .operator import AbstractOperator


class MoveOperator(AbstractOperator):

    name: str = "move"

    def __init__(
        self,
        particles: list[str],
        max_disp: float = 2.0,
        *args,
        **kwargs,
    ) -> None:
        """Initialise a MC move operator.
        
        Args:
            particles: The particles that can move.
            max_disp: The maximum displacement in [Ang].

        """
        super().__init__(
            *args,
            **kwargs,
        )

        self.particles = particles

        self.max_disp = max_disp

        return

    def run(self, atoms: Atoms, rng=np.random.default_rng()) -> Optional[Atoms]:
        """"""
        # Check species in the region
        super().run(atoms)
        self._extra_info = "-"

        # We need covalent bond distanes for neighbour check
        assert hasattr(self, "bond_distance_dict")

        # BUG: If there is no species in the system...
        species_indices = self._select_species(atoms, self.particles, rng=rng)

        # Get some basic stuff
        new_atoms = copy.deepcopy(atoms)

        # Initialise the neighbour list
        nl = NeighborList(
            self.covalent_max * np.array(natural_cutoffs(new_atoms)),
            skin=0.0,
            self_interaction=False,
            bothways=True,
        )

        # Find tag atoms
        # record original position of species_indices
        species = new_atoms[species_indices]
        assert isinstance(species, Atoms)
        self._extra_info = f"Move_{species.get_chemical_formula()}_{species_indices}"

        # TODO: Deal with pbc for molecules
        org_cop = np.mean(species.positions, axis=0)
        org_positions = species.positions.copy()

        # Move the species and use neighbour list to check atomic distances
        for i in range(self.MAX_RANDOM_ATTEMPTS):
            rvec = get_a_random_direction(rng)
            ran_pos = org_cop + rvec * self.max_disp
            species_ = copy.deepcopy(species)
            species_ = translate_then_rotate(
                species_, position=ran_pos, use_com=False, rng=rng
            )
            new_atoms.positions[species_indices] = species_.positions.copy()
            if check_atomic_distances_by_neighbour_list(
                new_atoms,
                neighlist=nl,
                atomic_indices=species_indices,
                covalent_ratio=[self.covalent_min, self.covalent_max],
                bond_distance_dict=self.bond_distance_dict,  # type: ignore
                allow_isolated=False,
            ):
                self._print(f"succeed to random after {i+1} attempts...")
                self._print("before pos: " + ("{:>12.4f} " * 3).format(*org_cop))
                self._print("random pos: " + ("{:>12.4f} " * 3).format(*ran_pos))
                new_cop = np.average(new_atoms.positions[species_indices], axis=0)
                self._print("actual pos: " + ("{:>12.4f} " * 3).format(*new_cop))
                break
            # Move failed and fallback to the original positions
            new_atoms.positions[species_indices] = org_positions
        else:
            new_atoms = None

        return new_atoms

    def metropolis(self, prev_ene: float, curr_ene: float, rng: np.random.Generator=np.random.default_rng()) -> bool:
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
