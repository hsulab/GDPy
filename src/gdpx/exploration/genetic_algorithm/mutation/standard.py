"""GDPy implementations of the standard ASE-GA mutation interfaces.

The public constructor names and configuration arguments intentionally follow
ASE-GA 1.0.3 so existing GDPy input files continue to work. The algorithms are
implemented in GDPy, and randomness is supplied exclusively by an explicit
:class:`numpy.random.Generator`.
"""

from __future__ import annotations

import copy

import numpy as np
from ase import Atoms

from gdpx.structures.geometry.ga import CellBounds, atoms_too_close

from ..core import OffspringCreator


def _movable_groups(atoms: Atoms, n_top: int, use_tags: bool) -> list[np.ndarray]:
    first = len(atoms) - n_top
    if not use_tags:
        return [np.array([index], dtype=int) for index in range(first, len(atoms))]
    tags = atoms.get_tags()
    return [np.flatnonzero(tags == tag) for tag in np.unique(tags[first:]) if tag != 0]


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
        groups = _movable_groups(atoms, self.n_top, self.use_tags)
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


class SoftMutation(OffspringCreator):
    """Displace movable atoms along a smooth, low-frequency random mode.

    This preserves the established ``soft`` configuration surface while using
    a lightweight internal mode generator instead of ASE-GA's global RNG path.
    """

    descriptor = "SoftMutation"
    min_inputs = 1

    def __init__(
        self,
        blmin,
        bounds=(0.5, 2.0),
        calculator=None,
        rcut=10.0,
        used_modes_file="used_modes.json",
        use_tags=False,
        verbose=False,
        rng=None,
    ):
        super().__init__(verbose=verbose, rng=rng)
        self.blmin = blmin
        self.bounds = bounds
        self.calculator = calculator
        self.rcut = rcut
        self.used_modes_file = used_modes_file
        self.use_tags = use_tags

    def mutate(self, atoms: Atoms):
        groups = _movable_groups(atoms, len(atoms), self.use_tags)
        if not groups:
            return None
        centers = np.array([atoms.positions[group].mean(axis=0) for group in groups])
        for _ in range(1000):
            phase = self.rng.normal(size=(len(groups), 3))
            if len(groups) > 1:
                distances = np.linalg.norm(centers[:, None] - centers[None, :], axis=2)
                kernel = np.exp(-distances / max(float(self.rcut), 1e-12))
                phase = kernel @ phase
            phase -= phase.mean(axis=0)
            scale = np.max(np.linalg.norm(phase, axis=1))
            if scale == 0:
                continue
            amplitude = self.rng.uniform(float(self.bounds[0]), float(self.bounds[1]))
            mutant = copy.deepcopy(atoms)
            for group, displacement in zip(groups, phase * amplitude / scale):
                mutant.positions[group] += displacement
            if not atoms_too_close(mutant, self.blmin, use_tags=self.use_tags):
                return mutant
        return None

    def get_new_individual(self, parents):
        child = self.mutate(parents[0])
        if child is None:
            return None, "mutation: soft"
        child = self.initialize_individual(parents[0], child)
        child.info["data"]["parents"] = [parents[0].info["confid"]]
        return self.finalize_individual(child), "mutation: soft"
