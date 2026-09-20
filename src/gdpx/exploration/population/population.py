"""Retained candidate references shared by all population search methods."""
from types import MappingProxyType

import numpy as np


def compute_population_fitness(structures, similarity_counts=None):
    """Score-based fitness with optional similarity penalties, without RNG or mutation."""
    if not structures:
        return np.empty(0)
    scores = np.asarray([a.info["key_value_pairs"]["raw_score"] for a in structures], dtype=float)
    if not np.isfinite(scores).all():
        raise ValueError("Population scores must be finite.")
    span = scores.min() - scores.max()
    fitness = np.ones(len(scores)) if span == 0 else 0.5 * (1 - np.tanh(2 * (scores - scores.max()) / span - 1))
    if similarity_counts is not None:
        fitness /= np.sqrt([1 + similarity_counts.get(a.info["confid"], 0) for a in structures])
    return fitness


def looks_like(first, second, comparator):
    if not np.array_equal(np.sort(first.numbers), np.sort(second.numbers)):
        return False
    return comparator.looks_like(first, second)


def count_looks_like(candidate, history, comparator):
    return sum(
        other.info["confid"] != candidate.info["confid"] and looks_like(candidate, other, comparator)
        for other in history
    )


def cache_fingerprints(structures, comparator):
    if hasattr(comparator, "_precompute_fingerprint"):
        comparator._precompute_fingerprint(structures)


def delete_fingerprints(structures, comparator):
    if hasattr(comparator, "_delete_fingerprint"):
        comparator._delete_fingerprint(structures)


class Population:
    """Rank and retain borrowed candidates; generation and selection live elsewhere."""

    def __init__(self, retained_size, comparator, use_extinct=False):
        if not isinstance(retained_size, int) or isinstance(retained_size, bool) or retained_size <= 0:
            raise ValueError("retained_size must be a positive integer.")
        self.retained_size = retained_size
        self.comparator = comparator
        self.use_extinct = use_extinct
        self._candidates = ()
        self.similarity_counts = MappingProxyType({})

    @property
    def candidates(self):
        """Immutable membership containing borrowed, unmodified Atoms references."""
        return self._candidates

    def refresh(self, database, *, history=None):
        """Rebuild membership and statistics from eligible relaxed search history."""
        history = sorted(
            database.get_all_relaxed_candidates(use_extinct=self.use_extinct) if history is None else history,
            key=lambda a: a.info["key_value_pairs"]["raw_score"],
            reverse=True,
        )
        selected = []
        similarity_counts = {}
        comparator = self.comparator
        # Comparators cache fingerprints in info. Preserve the original mapping
        # and its values by reference, including pre-existing caches; do not copy
        # atomic arrays, calculators, or nested metadata.
        original_info = [(atoms, atoms.info, dict(atoms.info)) for atoms in history]
        chunk_size = self.retained_size * 2
        try:
            for offset in range(0, len(history), chunk_size):
                chunk = history[offset:offset + chunk_size]
                cache_fingerprints(chunk, comparator)
                for candidate in chunk:
                    if len(selected) < self.retained_size and not any(
                        looks_like(candidate, other, comparator) for other in selected
                    ):
                        selected.append(candidate)
                delete_fingerprints([a for a in chunk if not any(a is s for s in selected)], comparator)
                if len(selected) == self.retained_size:
                    break
            similarity_counts = {candidate.info["confid"]: 0 for candidate in selected}
            for offset in range(0, len(history), chunk_size):
                chunk = history[offset:offset + chunk_size]
                cache_fingerprints(chunk, comparator)
                for candidate in selected:
                    similarity_counts[candidate.info["confid"]] += count_looks_like(candidate, chunk, comparator)
                delete_fingerprints([a for a in chunk if not any(a is s for s in selected)], comparator)
        finally:
            try:
                delete_fingerprints(history, comparator)
            finally:
                for atoms, info, values in original_info:
                    info.clear()
                    info.update(values)
                    atoms.info = info
        # Publish a complete snapshot only after successful comparison/cleanup.
        self._candidates = tuple(selected)
        self.similarity_counts = MappingProxyType(similarity_counts)
