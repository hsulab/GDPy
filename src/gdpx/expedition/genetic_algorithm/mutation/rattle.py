#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import copy
from typing import List

import numpy as np

from ase import Atoms
from ase.io import read, write
from ase.ga.offspring_creator import CombinationMutation, OffspringCreator

# from ..group import create_a_group
# from ..utils import check_overlap_neighbour


"""Some mutations for buffer atoms in the surface structure search."""


class RattleBufferMutation(OffspringCreator):

    #: Maximum number of attempts to rattle atoms.
    MAX_ATTEMPTS: int = 1000

    def __init__(self, group, nsel, strength=1.0, maxdisp=2.0, covalent_ratio=[0.8, 2.0],num_muts=1, rng=np.random, *args, **kwargs):
        """Initialize the RattleBufferMutation class.

        Args:
            group: Atom group to mutate.
            strength: Rattle strength.
            maxdisp: Maximum displacement.

        """
        super().__init__(num_muts=num_muts, rng=rng, *args, **kwargs)
        self.descriptor = "RattleBufferMutation"
        self.min_inputs = 1

        self.group = group
        self.nsel = nsel
        if self.nsel > 0:
            ...
        else:
            raise RuntimeError("`nsel` must be greater than zero.")

        self.strength = strength
        self.maxdisp = maxdisp

        self.covalent_ratio = covalent_ratio

        return

    def get_new_individual(self, parents: List[Atoms]):
        """"""
        f = parents[0]

        indi = self.mutate(f)
        if indi is None:
            return indi, "mutation: rattle_buffer"

        indi = self.initialize_individual(f, indi)
        indi.info["data"]["parents"] = [f.info["confid"]]
        group_indices = indi.info.pop("rattle_buffer_group", [])
        group_text = " ".join([str(i) for i in group_indices])

        return self.finalize_individual(indi), f"mutation: rattle_buffer {group_text}"

    def mutate(self, atoms: Atoms):
        """"""
        mutant = copy.deepcopy(atoms)

        group_indices = create_a_group(mutant, self.group)
        num_selected = 0
        if self.nsel < 1.0:
            num_selected = int(np.ceil(len(group_indices)*self.nsel))
        else:
            num_selected = int(self.nsel)
        group_indices = self.rng.choice(group_indices, size=num_selected, replace=False)  # TODO: change prob based on number of atoms per type?
        mutant.info["rattle_buffer_group"] = group_indices

        prev_positions = mutant.get_positions()[group_indices]
        # print(f"{prev_positions =}")

        for _ in range(self.MAX_ATTEMPTS):
            num_atoms = len(group_indices)
            pos_drift = self.strength*self.rng.uniform(low=-1.0, high=1.0, size=(num_atoms, 3))
            pos_drift_norm = np.linalg.norm(pos_drift, axis=1)[:, np.newaxis]
            pos_drift = pos_drift / pos_drift_norm
            pos_drift_norm = np.where(pos_drift_norm < self.maxdisp, pos_drift_norm, self.maxdisp)
            pos_drift = pos_drift*pos_drift_norm
            # print(f"{pos_drift =}")
            mutant.positions[group_indices] = prev_positions + pos_drift
            if check_overlap_neighbour(mutant, self.covalent_ratio):
                break
        else:
            mutant = None

        return mutant


if __name__ == "__main__":
    ...
  
