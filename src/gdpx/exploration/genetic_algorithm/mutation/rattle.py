"""ASE-GA-compatible rattle mutation implemented by GDPy."""

from __future__ import annotations

import numpy as np
from ase import Atoms

from gdpx.structures.geometry.ga import atoms_too_close

from ..core import OffspringCreator
from .utils import movable_groups


class RattleMutation(OffspringCreator):
    """Randomly translate atoms or tagged fragments without creating clashes."""

    descriptor = "RattleMutation"
    min_inputs = 1

    def __init__(
        self,
        blmin,
        n_top,
        rattle_strength=0.8,
        rattle_prop=0.4,
        test_dist_to_slab=True,
        use_tags=False,
        verbose=False,
        rng=None,
    ):
        super().__init__(verbose=verbose, rng=rng)
        self.blmin = blmin
        self.n_top = n_top
        self.rattle_strength = rattle_strength
        self.rattle_prop = rattle_prop
        self.test_dist_to_slab = test_dist_to_slab
        self.use_tags = use_tags

    def mutate(self, atoms: Atoms):
        groups = movable_groups(atoms, self.n_top, self.use_tags)
        if not groups:
            return None
        selected = [group for group in groups if self.rng.random() < self.rattle_prop]
        if not selected:
            selected = [groups[int(self.rng.integers(len(groups)))]]
        for _ in range(1000):
            mutant = atoms.copy()
            for group in selected:
                direction = self.rng.normal(size=3)
                norm = np.linalg.norm(direction)
                if norm == 0:
                    continue
                radius = self.rattle_strength * self.rng.random() ** (1.0 / 3.0)
                mutant.positions[group] += direction * radius / norm
            if not atoms_too_close(mutant, self.blmin, use_tags=self.use_tags):
                return mutant
        return None

    def get_new_individual(self, parents):
        child = self.mutate(parents[0])
        if child is None:
            return None, "mutation: rattle"
        child = self.initialize_individual(parents[0], child)
        child.info["data"]["parents"] = [parents[0].info["confid"]]
        return self.finalize_individual(child), "mutation: rattle"
