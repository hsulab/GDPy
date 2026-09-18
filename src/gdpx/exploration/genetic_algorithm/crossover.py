"""GDPy-owned implementations of ASE-GA 1.0.3-style crossovers."""

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np
from ase import Atoms

from gdpx.structures.geometry.ga import CellBounds, atoms_too_close

from .core import OffspringCreator


def _unit_vector(rng) -> np.ndarray:
    vector = rng.normal(size=3)
    norm = np.linalg.norm(vector)
    return vector / norm if norm else np.array([1.0, 0.0, 0.0])


def _splice_positions(first: Atoms, second: Atoms, rng, use_tags: bool) -> Atoms:
    """Combine parent groups while retaining the first parent's composition."""
    center = 0.5 * (first.positions.mean(axis=0) + second.positions.mean(axis=0))
    normal = _unit_vector(rng)

    def groups(atoms):
        if not use_tags:
            return [np.array([i], dtype=int) for i in range(len(atoms))]
        return [np.flatnonzero(atoms.get_tags() == tag) for tag in np.unique(atoms.get_tags())]

    required = Counter(tuple(first.numbers[group]) for group in groups(first))
    candidates = defaultdict(list)
    for parent_index, atoms in enumerate((first, second)):
        sign = -1.0 if parent_index == 0 else 1.0
        for group in groups(atoms):
            key = tuple(atoms.numbers[group])
            score = sign * float(np.dot(atoms.positions[group].mean(axis=0) - center, normal))
            candidates[key].append((score, rng.random(), atoms[group]))

    pieces = []
    for key, amount in required.items():
        choices = sorted(candidates[key], key=lambda item: (item[0], item[1]), reverse=True)
        if len(choices) < amount:
            return first.copy()
        pieces.extend(choice[2] for choice in choices[:amount])
    child = Atoms(cell=first.cell, pbc=first.pbc)
    next_tag = 1
    for piece in pieces:
        copy_piece = piece.copy()
        if use_tags:
            copy_piece.set_tags(next_tag)
            next_tag += 1
        child += copy_piece
    return child


class CutSpliceCrossover(OffspringCreator):
    """Cut-and-splice crossover for unsupported clusters."""

    descriptor = "CutSpliceCrossover"
    min_inputs = 2

    def __init__(self, blmin, keep_composition=True, rng=None):
        super().__init__(rng=rng)
        self.blmin = blmin
        self.keep_composition = keep_composition
        self.allow_variable_composition = not keep_composition

    def cross(self, first: Atoms, second: Atoms):
        for _ in range(1000):
            child = _splice_positions(first, second, self.rng, use_tags=False)
            if not atoms_too_close(child, self.blmin):
                return child
        return None

    def get_new_individual(self, parents):
        child = self.cross(parents[0], parents[1])
        if child is None:
            return None, "pairing: cut_and_splice_cluster"
        child = self.initialize_individual(parents[0], child)
        child.info["data"]["parents"] = [parent.info["confid"] for parent in parents]
        return self.finalize_individual(child), "pairing: cut_and_splice_cluster"


class CutAndSplicePairing(OffspringCreator):
    """Cut-and-splice pairing for surfaces and variable-cell structures."""

    descriptor = "CutAndSplicePairing"
    min_inputs = 2

    def __init__(
        self,
        slab,
        n_top,
        blmin,
        number_of_variable_cell_vectors=0,
        p1=1,
        p2=0.05,
        minfrac=None,
        cellbounds=None,
        test_dist_to_slab=True,
        use_tags=False,
        rng=None,
        verbose=False,
    ):
        super().__init__(verbose=verbose, rng=rng)
        self.slab = slab.copy()
        self.n_top = n_top
        self.blmin = blmin
        self.number_of_variable_cell_vectors = number_of_variable_cell_vectors
        self.p1 = p1
        self.p2 = p2
        self.minfrac = minfrac
        self.cellbounds = CellBounds() if cellbounds is None else cellbounds
        self.test_dist_to_slab = test_dist_to_slab
        self.use_tags = use_tags
        self.allow_variable_composition = False
        self.scaling_volume = None

    def update_scaling_volume(self, population, w_adapt=0.5, n_adapt=0):
        volumes = [atoms.get_volume() for atoms in population if atoms.get_volume() > 0]
        if volumes:
            target = float(np.median(volumes[-max(1, n_adapt or len(volumes)) :]))
            self.scaling_volume = target if self.scaling_volume is None else (
                w_adapt * target + (1.0 - w_adapt) * self.scaling_volume
            )

    def _variable_cell(self, first: Atoms, second: Atoms) -> np.ndarray:
        nvar = self.number_of_variable_cell_vectors
        if nvar <= 0:
            return first.cell.array.copy()
        mix = self.rng.uniform(0.25, 0.75)
        cell = first.cell.array.copy()
        cell[:nvar] = mix * first.cell.array[:nvar] + (1.0 - mix) * second.cell.array[:nvar]
        if self.scaling_volume and np.linalg.det(cell) > 0:
            cell[:nvar] *= (self.scaling_volume / np.linalg.det(cell)) ** (1.0 / 3.0)
        return cell

    def cross(self, first: Atoms, second: Atoms):
        substrate_size = len(first) - self.n_top
        top_first = first[substrate_size:]
        top_second = second[substrate_size:]
        for _ in range(1000):
            top = _splice_positions(top_first, top_second, self.rng, self.use_tags)
            child = self.slab.copy() + top
            cell = self._variable_cell(first, second)
            if np.linalg.det(cell) <= 0 or not self.cellbounds.is_within_bounds(cell):
                continue
            child.set_cell(cell, scale_atoms=False)
            child.set_pbc(first.pbc)
            if not atoms_too_close(child, self.blmin, use_tags=self.use_tags):
                return child
        return None

    def get_new_individual(self, parents):
        child = self.cross(parents[0], parents[1])
        if child is None:
            return None, "pairing: cut_and_splice"
        child = self.initialize_individual(parents[0], child)
        child.info["data"]["parents"] = [parent.info["confid"] for parent in parents]
        return self.finalize_individual(child), "pairing: cut_and_splice"
