"""ASE-GA-compatible strain mutation implemented by GDPy."""

from __future__ import annotations

import numpy as np
from ase import Atoms
from ase.calculators.lammps.coordinatetransform import calc_rotated_cell
from ase.cell import Cell

from gdpx.structures.geometry.ga import CellBounds, atoms_too_close, gather_atoms_by_tag

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
        if not n_adapt:
            n_adapt = int(np.ceil(0.2 * len(population)))
        new_volume = np.mean([atoms.get_volume() for atoms in population[:n_adapt]])
        if not self.scaling_volume:
            self.scaling_volume = new_volume
        else:
            self.scaling_volume = np.average(
                [self.scaling_volume, new_volume], weights=[1.0 - w_adapt, w_adapt]
            )

    def mutate(self, atoms: Atoms):
        reference_cell = atoms.get_cell()
        reference_positions = atoms.get_positions()
        reference_volume = atoms.get_volume() if self.scaling_volume is None else self.scaling_volume

        if self.use_tags:
            tags = atoms.get_tags()
            gather_atoms_by_tag(atoms)
            gathered_positions = atoms.get_positions()

        mutant = atoms.copy()
        count = 0
        too_close = True
        while too_close and count < 1000:
            count += 1
            strain = np.identity(3)
            for i in range(self.number_of_variable_cell_vectors):
                for j in range(i + 1):
                    random_value = self.rng.normal(loc=0.0, scale=self.stddev)
                    if i == j:
                        strain[i, j] += random_value
                    else:
                        epsilon = 0.5 * random_value
                        strain[i, j] += epsilon
                        strain[j, i] += epsilon

            new_cell = calc_rotated_cell(np.dot(strain, reference_cell))
            for i in range(self.number_of_variable_cell_vectors, 3):
                new_cell[i] = reference_cell[i]
            new_cell = Cell(new_cell)

            if self.number_of_variable_cell_vectors > 0:
                scaling = reference_volume / new_cell.volume
                scaling **= 1.0 / self.number_of_variable_cell_vectors
                new_cell[: self.number_of_variable_cell_vectors] *= scaling
            if not self.cellbounds.is_within_bounds(new_cell):
                continue

            mutant.set_cell(reference_cell, scale_atoms=False)
            if self.use_tags:
                transformation = np.linalg.solve(reference_cell, new_cell)
                for tag in np.unique(tags):
                    selected = np.where(tags == tag)
                    center = np.mean(gathered_positions[selected], axis=0)
                    displacement = np.dot(center, transformation) - center
                    mutant.positions[selected] += displacement
            else:
                mutant.set_positions(reference_positions)
            mutant.set_cell(new_cell, scale_atoms=not self.use_tags)
            mutant.wrap()
            too_close = atoms_too_close(mutant, self.blmin, use_tags=self.use_tags)

        return None if count == 1000 else mutant

    def get_new_individual(self, parents):
        child = self.mutate(parents[0])
        if child is None:
            return None, "mutation: strain"
        child = self.initialize_individual(parents[0], child)
        child.info["data"]["parents"] = [parents[0].info["confid"]]
        return self.finalize_individual(child), "mutation: strain"
