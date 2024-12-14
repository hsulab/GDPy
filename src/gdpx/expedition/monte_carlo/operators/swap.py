#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import copy
from typing import List, Optional

import numpy as np
from ase import Atoms
from ase.neighborlist import NeighborList, natural_cutoffs

from gdpx.geometry.particle import translate_then_rotate
from gdpx.geometry.spatial import check_atomic_distances_by_neighbour_list

from .move import MoveOperator


class SwapOperator(MoveOperator):

    name: str = "swap"

    def __init__(
        self,
        particles: List[str],
        *args,
        **kwargs,
    ):
        """"""
        super().__init__(
            particles=particles,
            *args,
            **kwargs,
        )

        # Prohibit swapping the same type of particles.
        if len(set(self.particles)) != 2:
            raise Exception(
                f"{self.__class__.__name__} needs two different types of particles."
            )

        return

    def run(
        self, atoms: Atoms, rng: np.random.Generator = np.random.default_rng()
    ) -> Optional[Atoms]:
        """"""
        # We only need check region without other in move_operator.
        self._check_region(atoms)
        self._extra_info = "-"

        # We need covalent bond distanes for neighbour check
        assert hasattr(self, "bond_distance_dict")

        # Build neighbour list
        new_atoms = copy.deepcopy(atoms)
        nl = NeighborList(
            self.covalent_max * np.array(natural_cutoffs(new_atoms)),
            skin=0.0,
            self_interaction=False,
            bothways=True,
        )

        # Swap the species
        for i in range(self.MAX_RANDOM_ATTEMPTS):
            # Get a new copy
            new_atoms = copy.deepcopy(atoms)

            # Pick an atom either index of an atom or tag of an moiety
            pick_one = self._select_species(new_atoms, [self.particles[0]], rng=rng)
            pick_two = self._select_species(new_atoms, [self.particles[1]], rng=rng)
            self._print(f"1->{pick_one} 2->{pick_two}")

            # Find particles by picked tags before swap
            particle_one = new_atoms[pick_one]  # default copy
            assert isinstance(particle_one, Atoms)
            particle_two = new_atoms[pick_two]
            assert isinstance(particle_two, Atoms)

            # TODO: Deal with pbc for molecules
            cop_one = copy.deepcopy(np.average(particle_one.get_positions(), axis=0))
            cop_two = copy.deepcopy(np.average(particle_two.get_positions(), axis=0))

            self._print(
                f"before: {particle_one.get_chemical_formula():>24s} "
                + ("{:>12.4f}" * 3).format(*cop_one)
            )
            self._print(
                f"before: {particle_two.get_chemical_formula():>24s} "
                + ("{:>12.4f}" * 3).format(*cop_two)
            )

            # Swap two positions with rotatation
            particle_one_ = translate_then_rotate(
                particle_one, position=cop_one, use_com=False, rng=rng
            )
            particle_two_ = translate_then_rotate(
                particle_two, position=cop_two, use_com=False, rng=rng
            )

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
                f"actual: {particle_one.get_chemical_formula():>24s} "
                + ("{:>12.4f}" * 3).format(*cop_one)
            )
            self._print(
                f"actual: {particle_two.get_chemical_formula():>24s} "
                + ("{:>12.4f}" * 3).format(*cop_two)
            )

            # Use neighbour list
            atomic_indices = [*pick_one, *pick_two]
            if check_atomic_distances_by_neighbour_list(
                new_atoms,
                neighlist=nl,
                atomic_indices=atomic_indices,
                covalent_ratio=[self.covalent_min, self.covalent_max],
                bond_distance_dict=self.bond_distance_dict,  # type: ignore
                allow_isolated=False,
            ):
                self._print(f"succeed to random after {i+1} attempts...")
                self._extra_info = f"S_{particle_one.get_chemical_formula()}_{pick_one}^{particle_two.get_chemical_formula()}_{pick_two}"
                break
        else:
            new_atoms = None
            self._extra_info = f"Swap_Failed"

        return new_atoms

    def as_dict(self) -> dict:
        """"""
        params = super().as_dict()
        params["particles"] = self.particles

        return params

    def __repr__(self) -> str:
        """"""
        content = f"@Modifier {self.__class__.__name__}\n"
        content += (
            f"temperature {self.temperature} [K] pressure {self.pressure} [bar]\n"
        )
        content += "covalent ratio: \n"
        content += f"  min: {self.covalent_min} max: {self.covalent_max}\n"
        content += f"swapped groups: \n"
        content += f"  {self.particles[0]} <-> {self.particles[1]}\n"

        return content


if __name__ == "__main__":
    ...
