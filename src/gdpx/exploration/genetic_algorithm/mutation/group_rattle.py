"""Rattle atoms selected by a GDPy group expression."""

from __future__ import annotations

import copy

import numpy as np
from ase import Atoms

from gdpx.structures.geometry.ga import atoms_too_close, closest_distances_generator
from gdpx.structures.groups import evaluate_group_expression

from ..core import OffspringCreator


class GroupRattleMutation(OffspringCreator):
    """Randomly displace a number or fraction of atoms in a selected group."""

    descriptor = "GroupRattleMutation"
    min_inputs = 1
    MAX_ATTEMPTS = 1000

    def __init__(
        self,
        group,
        nsel,
        strength=1.0,
        maxdisp=2.0,
        covalent_ratio=(0.8, 2.0),
        num_muts=1,
        use_tags=True,
        rng=None,
        *args,
        **kwargs,
    ):
        super().__init__(num_muts=num_muts, rng=rng, *args, **kwargs)
        if nsel <= 0:
            raise ValueError("nsel must be greater than zero.")
        self.group = group
        self.nsel = nsel
        self.strength = strength
        self.maxdisp = maxdisp
        self.covalent_ratio = covalent_ratio
        self.use_tags = use_tags

    def get_new_individual(self, parents):
        parent = parents[0]
        child = self.mutate(parent)
        if child is None:
            return None, "mutation: group_rattle"
        child = self.initialize_individual(parent, child)
        child.info["data"]["parents"] = [parent.info["confid"]]
        selected = child.info.pop("group_rattle_indices", [])
        selection = " ".join(str(index) for index in selected)
        return self.finalize_individual(child), f"mutation: group_rattle {selection}"

    def mutate(self, atoms: Atoms):
        candidate_indices = evaluate_group_expression(atoms, self.group)
        if not candidate_indices:
            return None
        if self.nsel < 1.0:
            selection_size = max(1, int(np.ceil(len(candidate_indices) * self.nsel)))
        else:
            selection_size = int(self.nsel)
        if selection_size > len(candidate_indices):
            raise ValueError(
                f"Cannot select {selection_size} atoms from group of size {len(candidate_indices)}."
            )
        selected = self.rng.choice(candidate_indices, size=selection_size, replace=False)
        minimum_distances = closest_distances_generator(
            atoms.numbers, float(self.covalent_ratio[0])
        )
        initial_positions = atoms.positions[selected].copy()
        for _ in range(self.MAX_ATTEMPTS):
            mutant = copy.deepcopy(atoms)
            directions = self.rng.normal(size=(selection_size, 3))
            norms = np.linalg.norm(directions, axis=1, keepdims=True)
            if np.any(norms == 0):
                continue
            amplitudes = np.minimum(
                self.rng.random((selection_size, 1)) * self.strength,
                self.maxdisp,
            )
            mutant.positions[selected] = initial_positions + directions / norms * amplitudes
            if not atoms_too_close(mutant, minimum_distances, use_tags=self.use_tags):
                mutant.info["group_rattle_indices"] = selected.tolist()
                return mutant
        return None
