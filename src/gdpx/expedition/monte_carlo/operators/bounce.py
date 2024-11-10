#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import copy
import functools
from typing import List, Optional

import numpy as np
from ase import Atoms, units
from ase.neighborlist import NeighborList, natural_cutoffs

from .. import bounce_one_atom
from .operator import AbstractOperator


class BounceOperator(AbstractOperator):

    name: str = "bounce"

    def __init__(
        self,
        particles: List[str],
        direction: str = "",
        max_disp: float = 2.0,
        *args,
        **kwargs,
    ):
        """"""
        super().__init__(*args, **kwargs)

        self.particles = particles
        self.direction = direction
        self.max_disp = max_disp

        self.nlist_prototype = functools.partial(
            NeighborList, skin=0.0, self_interaction=False, bothways=True
        )

        return

    def run(self, atoms: Atoms, rng) -> Atoms:
        """"""
        super().run(atoms)
        self._extra_info = "-"

        # BUG: If there is no species in the system...
        species_indices = self._select_species(atoms, self.particles, rng=rng)

        assert len(species_indices) == 1

        self._extra_info = (
            f"Bounce({self.direction})_{atoms[species_indices].get_chemical_formula()}_{species_indices}"
        )

        # get neighbour list
        curr_atoms = copy.deepcopy(atoms)
        nlist = self.nlist_prototype(
            self.covalent_max * np.array(natural_cutoffs(curr_atoms))
        )

        # bounce one atom
        atom_index = species_indices[0]
        new_atoms = bounce_one_atom(
            curr_atoms,
            atom_index,
            biased_direction=self.direction,
            max_disp=self.max_disp,
            nlist=nlist,
            bond_min_dict=self.blmin,
            rng=rng,
            print_func=self._print,
        )

        return new_atoms

    def metropolis(self, prev_ene: float, curr_ene: float, rng=np.random) -> bool:
        """"""
        # acceptance ratio
        kBT_eV = units.kB * self.temperature
        beta = 1.0 / kBT_eV  # 1/(kb*T), eV

        coef = 1.0
        ene_diff = curr_ene - prev_ene
        acc_ratio = np.min([1.0, coef * np.exp(-beta * (ene_diff))])

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
        params["direction"] = self.direction
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
        content += f"direction: {self.direction}\n"
        content += f"max disp: {self.max_disp}\n"
        content += f"particles: \n"
        content += f"  {self.particles}\n"

        return content


if __name__ == "__main__":
    ...
