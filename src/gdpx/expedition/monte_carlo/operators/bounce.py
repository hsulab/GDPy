#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import functools
from typing import Optional

import numpy as np
from ase import Atoms
from ase.neighborlist import NeighborList, natural_cutoffs

from gdpx.geometry.bounce import bounce_one_atom

from .operator import BaseMCOperator, metropolis_by_energy_difference


class BounceOperator(BaseMCOperator):

    name: str = "bounce"

    def __init__(
        self,
        particles: list[str],
        direction: str = "",
        max_disp: float = 2.0,
        repulsion_strength: float = 1.0,
        *args,
        **kwargs,
    ):
        """"""
        super().__init__(*args, **kwargs)

        self.particles = particles
        self.direction = direction
        self.max_disp = max_disp
        self.repulsion_strength = repulsion_strength

        self.nlist_prototype = functools.partial(NeighborList, skin=0.0, self_interaction=False, bothways=True)

        return

    def run(self, atoms: Atoms, rng: np.random.Generator = np.random.default_rng()) -> Optional[Atoms]:
        """"""
        # Check species in the region
        super().run(atoms)
        self._extra_info = "-"

        # We need covalent bond distanes for neighbour check
        assert hasattr(self, "bond_distance_dict")

        # Use the reference to avoid copying?
        self._atoms = atoms
        new_atoms = atoms

        # Check if the particles are in the atoms
        particle_indices = self._select_species(atoms, self.particles, rng=rng)
        if len(particle_indices) == 0:
            # Skip if no particles found
            self._extra_info = "Bounce_Skipped"
            return None

        # Initialise the neighbor list
        nlist = self.nlist_prototype(self.covalent_max * np.array(natural_cutoffs(new_atoms)))

        # Find tag atoms
        particle = new_atoms[particle_indices]
        assert isinstance(particle, Atoms)
        self._extra_info = f"Bounce({self.direction})_{particle.get_chemical_formula()}_{particle_indices}"

        # Bounce the particle
        atom_index = particle_indices[0]
        new_atoms, bounced = bounce_one_atom(
            new_atoms,
            atom_index,
            biased_direction=self.direction,
            max_disp=self.max_disp,
            strength=self.repulsion_strength,
            nlist=nlist,
            covalent_ratio=(self.covalent_min, self.covalent_max),
            bond_distance_dict=self.bond_distance_dict,  # type: ignore
            rng=rng,
            print_func=self._print,
        )

        # Save state for revert
        picked_indices = [b[0] for b in bounced]
        before_positions = np.array([b[1] for b in bounced])
        self._state = {
            "picked_indices": picked_indices,
            "before_positions": np.array(before_positions),
        }

        return new_atoms

    def revert_state(self, atoms: Atoms) -> Atoms:
        """Revert the state of atoms."""
        picked_indices = self._state.get("picked_indices")
        before_positions = self._state.get("before_positions")
        atoms.positions[picked_indices] = before_positions

        return atoms

    def metropolis(self, prev_ene: float, curr_ene: float, rng: np.random.Generator = np.random.default_rng()) -> bool:
        """"""
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
        params["direction"] = self.direction
        params["max_disp"] = self.max_disp

        return params

    def __repr__(self) -> str:
        """"""
        content = f"@Modifier {self.__class__.__name__}\n"
        content += f"temperature {self.temperature} [K] pressure {self.pressure} [bar]\n"
        content += "covalent ratio: \n"
        content += f"  min: {self.covalent_min} max: {self.covalent_max}\n"
        content += f"direction: {self.direction}\n"
        content += f"max disp: {self.max_disp}\n"
        content += f"particles: \n"
        content += f"  {self.particles}\n"

        # add indent
        content = self.indent + content.replace("\n", "\n" + self.indent)

        return content


if __name__ == "__main__":
    ...
