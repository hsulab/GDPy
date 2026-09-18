"""Geometry primitives used by GDPy's genetic algorithms.

The public behavior follows the corresponding ASE-GA 1.0.3 algorithms, with
an independent GDPy implementation and no ``ase.ga`` runtime dependency.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Mapping

import numpy as np
from ase import Atoms
from ase.data import covalent_radii
from ase.geometry.cell import cell_to_cellpar
from scipy.spatial.distance import cdist


def closest_distances_generator(
    atom_numbers: Iterable[int], ratio_of_covalent_radii: float
) -> dict[tuple[int, int], float]:
    """Build the symmetric minimum-distance mapping used by GA operators."""
    numbers = tuple(dict.fromkeys(int(number) for number in atom_numbers))
    return {
        (first, second): ratio_of_covalent_radii
        * (covalent_radii[first] + covalent_radii[second])
        for first in numbers
        for second in numbers
    }


def gather_atoms_by_tag(atoms: Atoms) -> None:
    """Move same-tag atoms into one minimum-image representation in place."""
    tags = atoms.get_tags()
    positions = atoms.get_positions()
    for tag in np.unique(tags):
        indices = np.flatnonzero(tags == tag)
        if len(indices) > 1:
            vectors = atoms.get_distances(indices[0], indices[1:], mic=True, vector=True)
            positions[indices[1:]] = positions[indices[0]] + vectors
    atoms.set_positions(positions)


def _periodic_images(pbc: np.ndarray):
    axes = [(-1, 0, 1) if periodic else (0,) for periodic in pbc]
    return itertools.product(*axes)


def atoms_too_close(
    atoms: Atoms,
    minimum_distances: Mapping[tuple[int, int], float],
    use_tags: bool = False,
) -> bool:
    """Return whether any permitted pair is closer than its threshold."""
    candidate = atoms.copy()
    if use_tags:
        gather_atoms_by_tag(candidate)
    positions = candidate.get_positions()
    numbers = candidate.get_atomic_numbers()
    tags = candidate.get_tags()
    cell = np.asarray(candidate.cell)
    for image in _periodic_images(candidate.pbc):
        shifted = positions + np.dot(np.asarray(image), cell)
        distances = cdist(positions, shifted)
        central = image == (0, 0, 0)
        for i, first in enumerate(numbers):
            for j, second in enumerate(numbers):
                if central and (i == j or (use_tags and tags[i] == tags[j])):
                    continue
                if distances[i, j] < minimum_distances[(int(first), int(second))]:
                    return True
    return False


def atoms_too_close_two_sets(
    first: Atoms,
    second: Atoms,
    minimum_distances: Mapping[tuple[int, int], float],
) -> bool:
    """Return whether atoms from two sets violate pair-distance thresholds."""
    if not np.array_equal(first.pbc, second.pbc) or not np.allclose(first.cell, second.cell):
        raise ValueError("The two atom sets must have identical cells and periodicity.")
    first_positions = first.get_positions()
    second_positions = second.get_positions()
    cell = np.asarray(first.cell)
    for image in _periodic_images(first.pbc):
        shifted = second_positions + np.dot(np.asarray(image), cell)
        distances = cdist(first_positions, shifted)
        for i, first_number in enumerate(first.numbers):
            for j, second_number in enumerate(second.numbers):
                if distances[i, j] < minimum_distances[(int(first_number), int(second_number))]:
                    return True
    return False


def get_rotation_matrix(axis, angle: float) -> np.ndarray:
    """Return the Rodrigues rotation matrix for a unit direction."""
    ux, uy, uz = np.asarray(axis, dtype=float)
    cosine, sine = np.cos(angle), np.sin(angle)
    return np.array(
        [
            [ux * ux * (1 - cosine) + cosine, ux * uy * (1 - cosine) - uz * sine, ux * uz * (1 - cosine) + uy * sine],
            [ux * uy * (1 - cosine) + uz * sine, uy * uy * (1 - cosine) + cosine, uy * uz * (1 - cosine) - ux * sine],
            [ux * uz * (1 - cosine) - uy * sine, uy * uz * (1 - cosine) + ux * sine, uz * uz * (1 - cosine) + cosine],
        ]
    )


def _nearest_neighbor_distance(atoms: Atoms, distances: np.ndarray) -> float:
    masked = np.where(distances > 1e-12, distances, np.inf)
    nearest = np.min(masked, axis=1)
    finite = nearest[np.isfinite(nearest)]
    return float(np.median(finite)) if len(finite) else 0.0


def get_nnmat(atoms: Atoms, mic: bool = False) -> np.ndarray:
    """Calculate the normalized nearest-neighbor matrix fingerprint."""
    cached = atoms.info.get("data", {}).get("nnmat")
    if cached is not None:
        return np.asarray(cached)
    elements = sorted(set(atoms.get_chemical_symbols()))
    element_indices = {element: index for index, element in enumerate(elements)}
    matrix = np.zeros((len(elements), len(elements)))
    distances = atoms.get_all_distances(mic=mic)
    cutoff = _nearest_neighbor_distance(atoms, distances) + 0.2
    for i, atom in enumerate(atoms):
        row = element_indices[atom.symbol]
        for neighbor in np.flatnonzero(distances[i] < cutoff):
            matrix[row, element_indices[atoms[int(neighbor)].symbol]] += 1
    for row, element in enumerate(elements):
        matrix[row] /= sum(atom.symbol == element for atom in atoms)
    return matrix.ravel()


def get_cell_angles_lengths(cell) -> dict[str, float]:
    """Return conventional cell lengths and angles plus plane angles."""
    values_array = cell_to_cellpar(cell)
    values_array[3:] *= np.pi / 180.0
    values = dict(zip(("a", "b", "c", "alpha", "beta", "gamma"), values_array))
    cell = np.asarray(cell)
    volume = abs(np.linalg.det(cell))
    for index, name in enumerate(("phi", "chi", "psi")):
        plane = np.linalg.norm(np.cross(cell[(index + 1) % 3], cell[(index + 2) % 3]))
        sine = np.clip(abs(volume / (plane * np.linalg.norm(cell[index]))), 0.0, 1.0)
        values[name] = float(np.arcsin(sine))
    return values


class CellBounds:
    """Bounds for cell-vector lengths and angles."""

    def __init__(self, bounds: Mapping[str, Iterable[float]] | None = None):
        self.bounds = {
            "alpha": [0.0, np.pi],
            "beta": [0.0, np.pi],
            "gamma": [0.0, np.pi],
            "phi": [0.0, np.pi],
            "chi": [0.0, np.pi],
            "psi": [0.0, np.pi],
            "a": [0.0, 1e6],
            "b": [0.0, 1e6],
            "c": [0.0, 1e6],
        }
        for name, configured_bound in (bounds or {}).items():
            if name not in self.bounds:
                raise ValueError(f"Unknown cell-bound parameter {name!r}.")
            bound = [float(value) for value in configured_bound]
            if len(bound) != 2 or bound[0] > bound[1]:
                raise ValueError(f"Cell bound {name!r} must contain an ordered pair.")
            if name not in {"a", "b", "c"}:
                bound = [value * np.pi / 180.0 for value in bound]
            self.bounds[name] = bound

    def is_within_bounds(self, cell) -> bool:
        values = get_cell_angles_lengths(cell)
        return all(lower <= values[name] <= upper for name, (lower, upper) in self.bounds.items())
