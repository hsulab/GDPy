"""ASE-GA-compatible strain mutation implemented by GDPy."""

from __future__ import annotations

import numpy as np
from ase import Atoms

from gdpx.structures.geometry.ga import CellBounds, atoms_too_close

from ..core import OffspringCreator


class StrainMutation(OffspringCreator):
    """Apply a symmetric random strain while respecting configured cell bounds."""

    descriptor = "StrainMutation"
    min_inputs = 1

    def __init__(
        self,
        blmin,
        cellbounds=None,
        stddev=0.7,
        number_of_variable_cell_vectors=3,
        use_tags=False,
        rng=None,
        verbose=False,
    ):
        super().__init__(verbose=verbose, rng=rng)
        self.blmin = blmin
        self.cellbounds = CellBounds() if cellbounds is None else cellbounds
        self.stddev = stddev
        self.number_of_variable_cell_vectors = number_of_variable_cell_vectors
        self.use_tags = use_tags
        self.scaling_volume = None

    def update_scaling_volume(self, population, w_adapt=0.5, n_adapt=0):
        volumes = [atoms.get_volume() for atoms in population if atoms.get_volume() > 0]
        if volumes:
            target = float(np.median(volumes[-max(1, n_adapt or len(volumes)) :]))
            self.scaling_volume = target if self.scaling_volume is None else (
                w_adapt * target + (1.0 - w_adapt) * self.scaling_volume
            )

    def mutate(self, atoms: Atoms):
        nvar = self.number_of_variable_cell_vectors
        if nvar <= 0:
            return None
        old_cell = atoms.cell.array.copy()
        for _ in range(1000):
            noise = self.rng.normal(0.0, self.stddev, size=(3, 3))
            strain = np.eye(3) + 0.5 * (noise + noise.T) / 3.0
            new_cell = old_cell.copy()
            new_cell[:nvar] = (old_cell @ strain)[:nvar]
            if np.linalg.det(new_cell) <= 0 or not self.cellbounds.is_within_bounds(new_cell):
                continue
            if self.scaling_volume:
                scale = (self.scaling_volume / abs(np.linalg.det(new_cell))) ** (1.0 / 3.0)
                new_cell[:nvar] *= scale
            mutant = atoms.copy()
            mutant.set_cell(new_cell, scale_atoms=True)
            if not atoms_too_close(mutant, self.blmin, use_tags=self.use_tags):
                return mutant
        return None

    def get_new_individual(self, parents):
        child = self.mutate(parents[0])
        if child is None:
            return None, "mutation: strain"
        child = self.initialize_individual(parents[0], child)
        child.info["data"]["parents"] = [parents[0].info["confid"]]
        return self.finalize_individual(child), "mutation: strain"
