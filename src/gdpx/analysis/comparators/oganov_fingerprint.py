"""Species-resolved Oganov fingerprints for structural comparison."""

from itertools import combinations_with_replacement

import numpy as np
from ase.neighborlist import NeighborList
from ase.utils import pbc2pbc
from scipy.spatial.distance import cdist
from scipy.special import erf

from .comparator import BaseComparator


class OganovFingerprintComparator(BaseComparator):
    """Compare selected atomic environments using weighted cosine distance.

    Prepared fingerprints are fresh, operation-owned data. Only geometric
    similarity is considered; compose with an energy comparator when needed.
    """

    name = "oganov_fingerprint"

    def __init__(self, group_expression=None, cos_dist_max=5e-3,
                 rcut=20.0, binwidth=0.05, sigma=0.02, nsigma=4,
                 pbc=True, maxdims=None, **kwargs):
        if "n_top" in kwargs:
            raise ValueError("n_top is removed; use group_expression to select comparison atoms.")
        if "dE" in kwargs:
            raise ValueError("dE is removed; use energetic_oganov_fingerprint with ene_diff_tol, or compose an energy comparator.")
        super().__init__(**kwargs)
        self.group_expression = group_expression
        self.cos_dist_max = cos_dist_max
        self.rcut = rcut
        self.binwidth = binwidth
        self.pbc = pbc2pbc(pbc)
        self.maxdims = [None] * 3 if maxdims is None else list(maxdims)
        self.sigma = sigma
        self.nsigma = nsigma
        self.dimensions = self.pbc.sum()
        if not np.isfinite(cos_dist_max) or not 0 < cos_dist_max <= 1:
            raise ValueError("cos_dist_max must be in (0, 1].")
        for name, value in dict(rcut=rcut, binwidth=binwidth, sigma=sigma, nsigma=nsigma).items():
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite.")
        if len(self.maxdims) != 3:
            raise ValueError("maxdims must contain three dimensions.")
        if self.dimensions in (1, 2):
            for direction in range(3):
                if not self.pbc[direction] and self.maxdims[direction] is not None:
                    if self.maxdims[direction] <= 0:
                        raise ValueError("Non-periodic maximum dimensions must be positive.")

    def prepare_data(self, structures):
        """Return fresh fingerprints of compatible, explicitly selected groups."""
        return [self._take_fingerprints(group) for group in self._prepare_groups(structures)]

    def fingerprint_distance(self, first, second):
        """Return geometric distance; no energy gate or input cache is used."""
        first_fp, first_types = first
        second_fp, second_types = second
        if (set(first_fp) != set(second_fp) or
                {n: len(indices) for n, indices in first_types.items()} !=
                {n: len(indices) for n, indices in second_types.items()}):
            raise ValueError("Comparison groups must have identical compositions.")
        if any(first_fp[key].shape != second_fp[key].shape for key in first_fp):
            raise ValueError("Fingerprint shapes differ; use the same comparator settings.")
        return float(self._cosine_distance(first_fp, second_fp, first_types))

    def compare_fingerprints(self, first, second):
        return self.fingerprint_distance(first, second) < self.cos_dist_max

    def looks_like(self, first, second):
        return self.compare_fingerprints(*self.prepare_data([first, second]))

    def _take_fingerprints(self, atoms, individual=False):
        positions = atoms.get_positions()
        numbers = atoms.get_atomic_numbers()
        cell = atoms.get_cell()
        unique_types = np.unique(numbers)
        type_indices = {}
        for atom_type in unique_types:
            type_indices[atom_type] = [i for i, atom in enumerate(atoms) if atom.number == atom_type]

        volume, pmin, pmax, qmin, qmax = self._get_volume(atoms)
        nonperiodic = [i for i in range(3) if not self.pbc[i]]

        def surface_0d(radius):
            return 4 * np.pi * radius**2

        def surface_2d(radius, position):
            coordinate = position[nonperiodic[0]]
            area = np.minimum(pmax - coordinate, radius) + np.minimum(coordinate - pmin, radius)
            return area * 2 * np.pi * radius

        def surface_1d(radius, position):
            coordinate = position[nonperiodic[1]]
            phi1 = np.lib.scimath.arccos((qmax - coordinate) / radius).real
            phi2 = np.pi - np.lib.scimath.arccos((qmin - coordinate) / radius).real
            return surface_2d(radius, position) * (1 - (phi1 + phi2) / np.pi)

        candidate = atoms.copy()
        candidate.set_pbc(self.pbc)
        neighbors = NeighborList(
            [self.rcut / 2.0] * len(candidate), skin=0.0, self_interaction=False, bothways=True
        )
        neighbors.update(candidate)

        margin = int(np.ceil(self.nsigma * self.sigma / self.binwidth))
        x = 0.25 * np.sqrt(2) * self.binwidth * (2 * margin + 1) / self.sigma
        smearing_norm = erf(x)
        number_of_bins = int(np.ceil(self.rcut / self.binwidth))
        bin_distances = self.binwidth * np.arange(1, number_of_bins + 1)

        def individual_rdf(index, atom_type):
            rdf = np.zeros(number_of_bins)
            if self.dimensions == 3:
                weights = 1.0 / surface_0d(bin_distances)
            elif self.dimensions == 2:
                weights = 1.0 / surface_2d(bin_distances, positions[index])
            elif self.dimensions == 1:
                weights = 1.0 / surface_1d(bin_distances, positions[index])
            else:
                weights = 1.0 / surface_0d(bin_distances)
            weights /= self.binwidth
            indices, offsets = neighbors.get_neighbors(index)
            valid_neighbors = np.where(numbers[indices] == atom_type)
            neighbor_positions = positions[indices[valid_neighbors]] + np.dot(offsets[valid_neighbors], cell)
            distances = cdist(neighbor_positions, [positions[index]])
            bins = np.floor(distances / self.binwidth)
            for offset in range(-margin, margin + 1):
                shifted_bins = bins + offset
                valid = np.where((shifted_bins >= 0) & (shifted_bins < number_of_bins))
                valid_bins = shifted_bins[valid].astype(int)
                values = weights[valid_bins]
                coefficient = 0.25 * np.sqrt(2) * self.binwidth / self.sigma
                values *= 0.5 * erf(coefficient * (2 * offset + 1)) - 0.5 * erf(
                    coefficient * (2 * offset - 1)
                )
                values /= smearing_norm
                for j, valid_bin in enumerate(valid_bins):
                    rdf[valid_bin] += values[j]
            rdf /= len(type_indices[atom_type]) / volume
            return rdf

        fingerprints = {}
        if individual:
            for index in range(len(atoms)):
                fingerprints[index] = {}
                for atom_type in unique_types:
                    fingerprint = individual_rdf(index, atom_type)
                    if self.dimensions > 0:
                        fingerprint -= 1
                    fingerprints[index][atom_type] = fingerprint
        else:
            for first_type, second_type in combinations_with_replacement(unique_types, 2):
                fingerprint = np.zeros(number_of_bins)
                for index in type_indices[first_type]:
                    fingerprint += individual_rdf(index, second_type)
                fingerprint /= len(type_indices[first_type])
                if self.dimensions > 0:
                    fingerprint -= 1
                fingerprints[(first_type, second_type)] = fingerprint
        return [fingerprints, type_indices]

    def _cosine_distance(self, first, second, type_indices):
        keys = sorted(first)
        weights = {
            key: len(type_indices[key[0]]) * len(type_indices[key[1]]) for key in keys
        }
        total = sum(weights.values())
        weights = {key: value / total for key, value in weights.items()}
        first_norm = np.sqrt(sum(np.linalg.norm(first[key]) ** 2 * weights[key] for key in keys))
        second_norm = np.sqrt(sum(np.linalg.norm(second[key]) ** 2 * weights[key] for key in keys))
        similarity = sum(
            np.sum(first[key] * second[key]) * weights[key] / (first_norm * second_norm)
            for key in keys
        )
        return 0.5 * (1 - similarity)

    def _get_volume(self, atoms):
        cell = atoms.get_cell()
        scaled = atoms.get_scaled_positions()
        volume = 1.0
        pmin = pmax = qmin = qmax = 0.0
        if self.dimensions in (1, 2):
            for direction in range(3):
                if not self.pbc[direction] and self.maxdims[direction] is None:
                    self.maxdims[direction] = np.linalg.norm(cell[direction])
        periodic = [i for i in range(3) if self.pbc[i]]
        nonperiodic = [i for i in range(3) if not self.pbc[i]]
        if self.dimensions == 3:
            volume = abs(np.dot(np.cross(cell[0], cell[1]), cell[2]))
        elif self.dimensions == 2:
            direction = nonperiodic[0]
            normal = np.cross(cell[periodic[0]], cell[periodic[1]])
            width = self.maxdims[direction] / np.linalg.norm(cell[direction])
            volume = abs(np.dot(normal, width * cell[direction]))
            span = np.ptp(scaled[:, direction])
            margin = 0.5 * (width - span)
            length = np.linalg.norm(cell[direction])
            pmin = (np.min(scaled[:, direction]) - margin) * length
            pmax = (np.max(scaled[:, direction]) + margin) * length
        elif self.dimensions == 1:
            periodic_direction = periodic[0]
            widths = [
                self.maxdims[direction] / np.linalg.norm(cell[direction]) for direction in nonperiodic
            ]
            volume = abs(
                np.dot(
                    np.cross(widths[0] * cell[nonperiodic[0]], widths[1] * cell[nonperiodic[1]]),
                    cell[periodic_direction],
                )
            )
            limits = []
            for direction, width in zip(nonperiodic, widths):
                margin = 0.5 * (width - np.ptp(scaled[:, direction]))
                length = np.linalg.norm(cell[direction])
                limits.extend(
                    [
                        (np.min(scaled[:, direction]) - margin) * length,
                        (np.max(scaled[:, direction]) + margin) * length,
                    ]
                )
            pmin, pmax, qmin, qmax = limits
        return [volume, pmin, pmax, qmin, qmax]
