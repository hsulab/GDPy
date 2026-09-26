"""ASE-GA-compatible rattle mutation implemented by GDPy."""

from __future__ import annotations

import numpy as np
from ase import Atoms

from gdpx.structures.geometry.ga import atoms_too_close, atoms_too_close_two_sets

from ..core import OffspringCreator
class RattleMutation(OffspringCreator):
    """Randomly translate atoms or tagged fragments without creating clashes."""

    descriptor = "RattleMutation"
    min_inputs = 1
    supports_fragment_preservation = True
    fragment_mode_configurable = True

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
        number_to_optimize = len(atoms) if self.n_top is None else self.n_top
        slab = atoms[: len(atoms) - number_to_optimize]
        top = atoms[-number_to_optimize:]
        tags = top.get_tags() if self.use_tags else np.arange(number_to_optimize)
        reference_positions = top.get_positions()
        displacement_width = 2.0 * self.rattle_strength

        attempt = 0
        too_close = True
        while too_close and attempt < 1000:
            attempt += 1
            positions = reference_positions.copy()
            moved = False
            for tag in np.unique(tags):
                selected = np.where(tags == tag)
                if self.rng.random() < self.rattle_prop:
                    moved = True
                    positions[selected] += displacement_width * (self.rng.random(3) - 0.5)
            if not moved:
                continue

            rattled_top = Atoms(
                top.numbers,
                positions=positions,
                cell=top.cell,
                pbc=top.pbc,
                tags=tags,
            )
            too_close = atoms_too_close(
                rattled_top,
                self.blmin,
                use_tags=self.use_tags,
            )
            if not too_close and self.test_dist_to_slab:
                too_close = atoms_too_close_two_sets(rattled_top, slab, self.blmin)

        if attempt == 1000:
            return None
        return slab + rattled_top

    def get_new_individual(self, parents):
        child = self.mutate(parents[0])
        if child is None:
            return None, "mutation: rattle"
        child = self.initialize_individual(parents[0], child)
        child.info["data"]["parents"] = [parents[0].info["confid"]]
        return self.finalize_individual(child), "mutation: rattle"
