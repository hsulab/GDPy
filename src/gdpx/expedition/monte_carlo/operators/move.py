#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import copy
from typing import Optional

import numpy as np
from ase import Atoms
from ase.neighborlist import NeighborList, natural_cutoffs

from gdpx.geometry.bounce import get_a_random_direction
from gdpx.geometry.particle import translate_then_rotate
from gdpx.geometry.spatial import check_atomic_distances_by_neighbour_list

from .operator import BaseMCOperator, metropolis_by_energy_difference


class MoveOperator(BaseMCOperator):

    name: str = "move"

    def __init__(
        self,
        particles: list[str],
        max_disp: float = 2.0,
        skip_distance_check: bool = False,
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

        self.skip_distance_check = skip_distance_check

        # Some state information after mc attempts and before energy evaluation
        self._atoms = None
        self._state = {}

        return

    def run(self, atoms: Atoms, rng=np.random.default_rng()) -> Optional[Atoms]:
        """"""
        # Check state
        assert self._state == {}, "State should be empty before running the operator."
        assert self._atoms is None, "Atoms should be None before running the operator."

        # Check species in the region
        super().run(atoms)
        self._extra_info = "-"

        # We need covalent bond distanes for neighbour check
        assert hasattr(self, "bond_distance_dict")

        # Check if the particles are in the atoms
        species_indices = self._select_species(atoms, self.particles, rng=rng)
        if len(species_indices) == 0:
            # Skip if no particles found
            return None

        # Use the reference to avoid copying?
        self._atoms = atoms
        new_atoms = atoms

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

        self._state = {
            "picked_indices": species_indices,
            "before_positions": org_positions,
        }

        # Move the species and use neighbour list to check atomic distances
        self._print(self.indent + f"check distance: {not self.skip_distance_check}")
        for i in range(self.MAX_RANDOM_ATTEMPTS):
            rvec = get_a_random_direction(rng)
            ran_pos = org_cop + rvec * self.max_disp
            species_ = copy.deepcopy(species)
            species_ = translate_then_rotate(species_, position=ran_pos, use_com=False, rng=rng)
            new_atoms.positions[species_indices] = species_.positions.copy()
            if self.skip_distance_check or check_atomic_distances_by_neighbour_list(
                new_atoms,
                neighlist=nl,
                atomic_indices=species_indices,
                covalent_ratio=[self.covalent_min, self.covalent_max],
                bond_distance_dict=self.bond_distance_dict,  # type: ignore
                allow_isolated=False,
            ):
                self._print(self.indent + f"succeed to random after {i+1} attempts...")
                self._print(self.indent + "before pos: " + ("{:>12.4f} " * 3).format(*org_cop))
                self._print(self.indent + "random pos: " + ("{:>12.4f} " * 3).format(*ran_pos))
                new_cop = np.average(new_atoms.positions[species_indices], axis=0)
                self._print(self.indent + "actual pos: " + ("{:>12.4f} " * 3).format(*new_cop))
                break
            # Move failed and fallback to the original positions
            new_atoms.positions[species_indices] = org_positions
        else:
            new_atoms = None

        return new_atoms

    def revert_state(self, atoms: Atoms) -> Atoms:
        """Revert the state of atoms."""
        picked_indices = self._state.get("picked_indices")
        before_positions = self._state.get("before_positions")

        atoms.positions[picked_indices] = before_positions

        return atoms

    def metropolis(self, prev_ene: float, curr_ene: float, rng: np.random.Generator = np.random.default_rng()) -> bool:
        """Metropolis criterion for the move operator."""
        success = metropolis_by_energy_difference(
            prev_ene=prev_ene,
            curr_ene=curr_ene,
            temperature=self.temperature,
            region=self.region,
            rng=rng,
            indent=self.indent,
            print_func=self._print,
        )

        if not success:
            assert self._atoms is not None, "Atoms should not be None when reverting state."
            self.revert_state(self._atoms)
        else:
            ...

        self._state = {}
        self._atoms = None

        return success

    def as_dict(self) -> dict:
        """"""
        params = super().as_dict()
        params["particles"] = self.particles
        params["max_disp"] = self.max_disp
        params["skip_distance_check"] = self.skip_distance_check

        return params

    def __repr__(self) -> str:
        """"""
        content = f"@Modifier {self.__class__.__name__}\n"
        content += f"temperature {self.temperature} [K] pressure {self.pressure} [bar]\n"
        content += "covalent ratio: \n"
        content += f"  min: {self.covalent_min} max: {self.covalent_max}\n"
        content += f"max disp: {self.max_disp}\n"
        content += f"particles: \n"
        content += f"  {self.particles}\n"

        # add indent
        content = self.indent + content.replace("\n", "\n" + self.indent)

        return content


if __name__ == "__main__":
    ...
