"""Ranked candidate references and fitness shared by population searches."""
import numpy as np


def compute_population_fitness(structures, with_history=True):
    if not structures:
        return []
    scores = np.asarray([a.info["key_value_pairs"]["raw_score"] for a in structures], dtype=float)
    if not np.isfinite(scores).all():
        raise ValueError("Population scores must be finite.")
    span = scores.min() - scores.max()
    fitness = np.ones(len(scores)) if span == 0 else 0.5 * (1 - np.tanh(2 * (scores - scores.max()) / span - 1))
    if with_history:
        fitness /= np.sqrt([1 + a.info.get("n_paired", 0) for a in structures])
        fitness /= np.sqrt([1 + a.info.get("looks_like", 0) for a in structures])
    return fitness.tolist()


def selection_weights(structures, with_history=True):
    fitness = np.asarray(compute_population_fitness(structures, with_history))
    return fitness / fitness.sum() if len(fitness) else fitness


def looks_like(first, second, comparator):
    if not np.array_equal(np.sort(first.numbers), np.sort(second.numbers)):
        return False
    return comparator.looks_like(first, second)


def count_looks_like(candidate, history, comparator):
    return sum(
        other.info["confid"] != candidate.info["confid"] and looks_like(candidate, other, comparator)
        for other in history
    )


def cache_fingerprints(structures, comparator, print_func=print):
    if hasattr(comparator, "_precompute_fingerprint"):
        comparator._precompute_fingerprint(structures)


def delete_fingerprints(structures, comparator, print_func=print):
    if hasattr(comparator, "_delete_fingerprint"):
        comparator._delete_fingerprint(structures)


class CandidatePool:
    """Own a list of borrowed candidates, never copies of their atomic arrays."""

    def __init__(self, database, retained_size, comparator, use_extinct=False):
        self.all_candidates = sorted(
            database.get_all_relaxed_candidates(use_extinct=use_extinct),
            key=lambda a: a.info["key_value_pairs"]["raw_score"],
            reverse=True,
        )
        self.candidates = []
        # Chunk precomputation bounds temporary fingerprint memory. Similarity
        # history still includes every eligible candidate, not only retained ones.
        chunk_size = max(1, retained_size * 2)
        try:
            for offset in range(0, len(self.all_candidates), chunk_size):
                chunk = self.all_candidates[offset:offset + chunk_size]
                cache_fingerprints(chunk, comparator)
                for candidate in chunk:
                    if len(self.candidates) < retained_size and not any(
                        looks_like(candidate, selected, comparator) for selected in self.candidates
                    ):
                        self.candidates.append(candidate)
                delete_fingerprints([a for a in chunk if not any(a is s for s in self.candidates)], comparator)
                if len(self.candidates) == retained_size:
                    break
            for candidate in self.candidates:
                candidate.info["looks_like"] = 0
                candidate.info["n_paired"] = 0
            for offset in range(0, len(self.all_candidates), chunk_size):
                chunk = self.all_candidates[offset:offset + chunk_size]
                cache_fingerprints(chunk, comparator)
                for candidate in self.candidates:
                    candidate.info["looks_like"] += count_looks_like(candidate, chunk, comparator)
                delete_fingerprints([a for a in chunk if not any(a is s for s in self.candidates)], comparator)
        finally:
            delete_fingerprints(self.all_candidates, comparator)

    def select(self, size, rng, with_history=True):
        if not self.candidates:
            return []
        indices = rng.choice(
            len(self.candidates), size=size, replace=True,
            p=selection_weights(self.candidates, with_history),
        )
        return [self.candidates[i] for i in indices]
