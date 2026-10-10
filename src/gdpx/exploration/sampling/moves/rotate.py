"""Rigid molecular rotations with reversible, particle-local coordinate edits."""

from numbers import Real

import numpy as np
from ase.data import atomic_numbers
from ase.geometry import find_mic
from ase.neighborlist import NeighborList, natural_cutoffs

from gdpx.structures.geometry.spatial import check_atomic_distances_by_neighbour_list

from .operator import BaseMCOperator


class RotateOperator(BaseMCOperator):
    """Rotate one tagged molecule about its center or a unique element anchor.

    The axis is isotropic unless specified; the signed angle is uniform in
    [-max_angle, max_angle] degrees. Periodic molecules are reconstructed in
    the minimum image before rotation, retaining each atom's original image.
    """

    name = "rotate"

    def __init__(self, particles, max_angle=180.0, center="com", axis=None, **kwargs):
        if (isinstance(max_angle, bool) or not isinstance(max_angle, Real) or
                not np.isfinite(max_angle) or not 0 < max_angle <= 180):
            raise ValueError("max_angle must be finite and in (0, 180] degrees.")
        if not isinstance(center, str) or center not in {"com", "cop", *atomic_numbers}:
            raise ValueError("center must be 'com', 'cop', or an element symbol.")
        if not isinstance(particles, (list, tuple)) or not particles or not all(
            isinstance(particle, str) and particle for particle in particles
        ):
            raise ValueError("particles must be a non-empty list of species.")
        if axis is not None:
            axis = np.asarray(axis, dtype=float)
            if (axis.shape != (3,) or not np.all(np.isfinite(axis)) or
                    not np.isfinite(np.linalg.norm(axis)) or not np.linalg.norm(axis) > 0):
                raise ValueError("axis must be a finite, nonzero three-vector.")
            axis = axis / np.linalg.norm(axis)
        super().__init__(**kwargs)
        self.particles = list(particles)
        self.max_angle = float(max_angle)
        self.center = center
        self.axis = None if axis is None else axis.tolist()

    def _propose(self, atoms, rng):
        super()._propose(atoms, rng)
        eligible_tags = {
            tag for species, tags in self._curr_tags_dict.items() if species in self.particles for tag in tags
        }
        groups = {tag: [] for tag in sorted(eligible_tags)}
        for index, tag in enumerate(atoms.get_tags()):
            if tag in groups:
                groups[tag].append(index)
        groups = [indices for indices in groups.values() if len(indices) > 1 and (
            self.center in ("com", "cop") or
            np.count_nonzero(atoms.numbers[indices] == atomic_numbers[self.center]) == 1
        )]
        if not groups:
            self._extra_info = "Rotate_Skipped"
            return None
        indices = groups[int(rng.choice(len(groups)))]
        original = atoms.positions[indices].copy()
        vectors, _ = find_mic(original - original[0], atoms.cell, atoms.pbc)
        if self.center == "com":
            pivot = np.average(vectors, axis=0, weights=atoms.get_masses()[indices])
        elif self.center == "cop":
            pivot = np.mean(vectors, axis=0)
        else:
            anchor = np.flatnonzero(atoms.numbers[indices] == atomic_numbers[self.center])[0]
            pivot = vectors[anchor]
        relative = vectors - pivot
        self._transaction.watch(indices)
        nl, distances = None, None
        if not self.skip_distance_check:
            nl = NeighborList(self.covalent_max * np.array(natural_cutoffs(atoms)),
                              skin=0.0, self_interaction=False, bothways=True)
            distances = dict(self.bond_distance_dict)
            distances.update(self.custom_pair_distance_dict or {})
        for _ in range(self.MAX_RANDOM_ATTEMPTS):
            axis = np.asarray(self.axis) if self.axis is not None else rng.normal(size=3)
            # A zero draw is vanishingly unlikely, but consumes an attempt safely.
            norm = np.linalg.norm(axis)
            if not norm > 0:
                continue
            axis = axis / norm
            angle = np.deg2rad(rng.uniform(-self.max_angle, self.max_angle))
            cosine, sine = np.cos(angle), np.sin(angle)
            rotated = (relative * cosine + np.cross(axis, relative) * sine +
                       np.outer(relative @ axis, axis) * (1 - cosine))
            atoms.positions[indices] = original + rotated - relative
            if self.center not in ("com", "cop"):
                atoms.positions[indices[anchor]] = original[anchor]
            if self.skip_distance_check or check_atomic_distances_by_neighbour_list(
                atoms, neighlist=nl, atomic_indices=indices, bond_distance_dict=distances,
                covalent_ratio=(self.covalent_min, self.covalent_max), allow_isolated=self.allow_isolated,
            ):
                self._extra_info = f"Rotate_{atoms[indices].get_chemical_formula()}_{indices}"
                return atoms
            atoms.positions[indices] = original
        self._extra_info = "Rotate_Failed"
        return None

    def as_dict(self):
        return dict(super().as_dict(), particles=list(self.particles), max_angle=self.max_angle,
                    center=self.center, axis=None if self.axis is None else list(self.axis))
