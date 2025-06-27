#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import copy
from typing import Optional

import numpy as np
from ase import Atoms
from ase.neighborlist import NeighborList, natural_cutoffs

from gdpx.geometry.particle import translate_then_rotate
from gdpx.geometry.spatial import check_atomic_distances_by_neighbour_list

from .operator import BaseMCOperator, metropolis_by_energy_difference


class SwapOperator(BaseMCOperator):

    name: str = "swap"

    def __init__(
        self,
        particles: list[str],
        *args,
        **kwargs,
    ):
        """"""
        super().__init__(
            *args,
            **kwargs,
        )

        self.particles = particles

        # Prohibit swapping the same type of particles.
        if len(set(self.particles)) != 2:
            raise Exception(f"{self.__class__.__name__} needs two different types of particles.")

        return

    def run(self, atoms: Atoms, rng: np.random.Generator = np.random.default_rng()) -> Optional[Atoms]:
        """"""
        # Check particles in the region
        super().run(atoms)
        self._extra_info = "-"

        # We need covalent bond distanes for neighbour check
        assert hasattr(self, "bond_distance_dict")

        # Use the reference to avoid copying?
        self._atoms = atoms
        new_atoms = atoms

        # Build neighbour list
        nl = NeighborList(
            self.covalent_max * np.array(natural_cutoffs(new_atoms)),
            skin=0.0,
            self_interaction=False,
            bothways=True,
        )

        # Swap the particles
        self._print(self.indent + f"check distance: {not self.skip_distance_check}")
        for i in range(self.MAX_RANDOM_ATTEMPTS):
            # Pick an atom either index of an atom or tag of an moiety
            pick_one = self._select_species(new_atoms, [self.particles[0]], rng=rng)
            pick_two = self._select_species(new_atoms, [self.particles[1]], rng=rng)
            self._print(self.indent + f"1->{pick_one} 2->{pick_two}")

            # Find particles by picked tags before swap
            particle_one = new_atoms[pick_one]  # default copy
            assert isinstance(particle_one, Atoms)
            positions_one = particle_one.get_positions()

            particle_two = new_atoms[pick_two]
            assert isinstance(particle_two, Atoms)
            positions_two = particle_two.get_positions()

            self._state = {
                "pick_one": pick_one,
                "pick_two": pick_two,
                "positions_one": positions_one,
                "positions_two": positions_two,
            }

            # TODO: Deal with pbc for molecules
            cop_one = copy.deepcopy(np.average(positions_one, axis=0))
            cop_two = copy.deepcopy(np.average(positions_two, axis=0))

            self._print(
                self.indent
                + f"before: {particle_one.get_chemical_formula():>24s} "
                + ("{:>12.4f}" * 3).format(*cop_one)
            )
            self._print(
                self.indent
                + f"before: {particle_two.get_chemical_formula():>24s} "
                + ("{:>12.4f}" * 3).format(*cop_two)
            )

            # Swap two positions with rotatation
            # TODO: how about velocity, charge, and magnetic moment?
            particle_one_ = translate_then_rotate(particle_one, position=cop_one, use_com=False, rng=rng)
            particle_two_ = translate_then_rotate(particle_two, position=cop_two, use_com=False, rng=rng)

            new_atoms.positions[pick_one] = particle_two_.positions
            new_atoms.positions[pick_two] = particle_one_.positions

            # Find particles by picked tags after swap
            particle_one = new_atoms[pick_one]  # default copy
            assert isinstance(particle_one, Atoms)
            particle_two = new_atoms[pick_two]
            assert isinstance(particle_two, Atoms)

            # TODO: Deal with pbc for molecules
            cop_one = copy.deepcopy(np.average(particle_one.get_positions(), axis=0))
            cop_two = copy.deepcopy(np.average(particle_two.get_positions(), axis=0))

            self._print(
                self.indent
                + f"actual: {particle_one.get_chemical_formula():>24s} "
                + ("{:>12.4f}" * 3).format(*cop_one)
            )
            self._print(
                self.indent
                + f"actual: {particle_two.get_chemical_formula():>24s} "
                + ("{:>12.4f}" * 3).format(*cop_two)
            )

            # Use neighbour list
            atomic_indices = [*pick_one, *pick_two]
            if self.skip_distance_check or check_atomic_distances_by_neighbour_list(
                new_atoms,
                neighlist=nl,
                atomic_indices=atomic_indices,
                covalent_ratio=[self.covalent_min, self.covalent_max],
                bond_distance_dict=self.bond_distance_dict,  # type: ignore
                allow_isolated=False,
            ):
                self._print(self.indent + f"succeed to random after {i+1} attempts...")
                self._extra_info = f"S_{particle_one.get_chemical_formula()}_{pick_one}^{particle_two.get_chemical_formula()}_{pick_two}"
                break
            else:
                # restore original positions
                new_atoms.positions[pick_one] = positions_one
                new_atoms.positions[pick_two] = positions_two
        else:
            new_atoms = None
            self._extra_info = f"Swap_Failed"

        return new_atoms

    def revert_state(self, atoms: Atoms) -> Atoms:
        """Revert the state of atoms."""
        pick_one = self._state.get("pick_one")
        pick_two = self._state.get("pick_two")
        positions_one = self._state.get("positions_one")
        positions_two = self._state.get("positions_two")

        atoms.positions[pick_one] = positions_one
        atoms.positions[pick_two] = positions_two

        return atoms

    def metropolis(self, prev_ene: float, curr_ene: float, rng: np.random.Generator = np.random.default_rng()) -> bool:
        """Metropolis criterion for the swap operator."""
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

        return params

    def __repr__(self) -> str:
        """"""
        content = f"@Modifier {self.__class__.__name__}\n"
        content += f"temperature {self.temperature} [K] pressure {self.pressure} [bar]\n"
        content += "covalent ratio: \n"
        content += f"  min: {self.covalent_min} max: {self.covalent_max}\n"
        content += f"swapped groups: \n"
        content += f"  {self.particles[0]} <-> {self.particles[1]}\n"

        # add indent
        content = self.indent + content.replace("\n", "\n" + self.indent)

        return content


if __name__ == "__main__":
    ...
