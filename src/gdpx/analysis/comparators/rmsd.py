"""Species-preserving assignment and rotation fitting for finite clusters."""

from itertools import permutations, product

import numpy as np
from scipy.optimize import linear_sum_assignment

from .comparator import BaseComparator


class PermutationRMSDComparator(BaseComparator):
    """Compare selected, contiguous clusters using species-preserving fitted RMSD."""

    name = "permutation_rmsd"

    def __init__(self, group_expression=None, rmsd_max=0.5, **kwargs):
        if "n_top" in kwargs:
            raise ValueError("n_top is removed; use group_expression to select comparison atoms.")
        super().__init__(**kwargs)
        if not np.isfinite(rmsd_max) or rmsd_max <= 0:
            raise ValueError("rmsd_max must be positive and finite.")
        self.group_expression = group_expression
        self.rmsd_max = rmsd_max

    def prepare_data(self, structures):
        return self._prepare_groups(structures)

    def fingerprint_distance(self, first, second):
        if not np.array_equal(np.sort(first.numbers), np.sort(second.numbers)):
            raise ValueError("Comparison groups must have identical compositions.")
        return permutation_rmsd(first, second)

    def compare_fingerprints(self, first, second):
        return self.fingerprint_distance(first, second) < self.rmsd_max

    def looks_like(self, first, second):
        return self.compare_fingerprints(*self.prepare_data([first, second]))


def permutation_rmsd(first, second):
    """Estimate minimum RMSD with species assignments and proper rotations.

    Alternating Hungarian assignment and Kabsch fitting uses the identity and
    all 24 proper principal-axis orientations as starting points. Multiple
    starts reduce local assignment minima; this is not an exhaustive search
    over all atom permutations. Coordinates must describe contiguous clusters.
    """
    if not np.array_equal(np.sort(first.numbers), np.sort(second.numbers)):
        return float("inf")
    x = first.positions - first.positions.mean(axis=0)
    y = second.positions - second.positions.mean(axis=0)
    _, axes_x = np.linalg.eigh(x.T @ x)
    _, axes_y = np.linalg.eigh(y.T @ y)
    rotations = [np.eye(3)]
    for perm in permutations(range(3)):
        for signs in product((-1, 1), repeat=3):
            rotation = axes_x @ (np.eye(3)[:, perm] * signs) @ axes_y.T
            if np.linalg.det(rotation) > 0:
                rotations.append(rotation)
    groups = [(np.flatnonzero(first.numbers == n), np.flatnonzero(second.numbers == n))
              for n in np.unique(first.numbers)]
    best = float("inf")
    for rotation in rotations:
        z = x @ rotation
        previous = float("inf")
        for _ in range(100):
            assignment = np.empty(len(first), dtype=int)
            for left, right in groups:
                cost = np.sum((z[left, None, :] - y[right, :]) ** 2, axis=2)
                rows, columns = linear_sum_assignment(cost)
                assignment[left[rows]] = right[columns]
            u, _, vt = np.linalg.svd(x.T @ y[assignment])
            correction = np.eye(3)
            correction[-1, -1] = np.linalg.det(u @ vt)
            z = x @ (u @ correction @ vt)
            error = float(np.sqrt(np.sum((z - y[assignment]) ** 2) / len(first)))
            best = min(best, error)
            if abs(previous - error) < 1e-10:
                break
            previous = error
    return best
