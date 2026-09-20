"""GA parent selection, pairing history, and composition-group policies."""
import numpy as np

from ..population.population import compute_population_fitness


class GeneticParentSelector:
    def __init__(self, rng, composition="constant"):
        if composition not in ("constant", "variable"):
            raise ValueError("Population name must be `constant` or `variable`.")
        self.rng = rng
        self.composition = composition
        self.participation = {}
        self.groups = ()

    def refresh(self, population, database):
        """Refresh algorithm history and groups after population.refresh()."""
        participation, _ = database.get_participation_in_pairing()
        self.participation = dict(participation)
        groups = {}
        for candidate in population.candidates:
            # Keep the former grouping and declaration order: ordered species,
            # rather than a formula that merges different atomic arrangements.
            key = "".join(candidate.get_chemical_symbols())
            groups.setdefault(key, []).append(candidate)
        self.groups = tuple(tuple(group) for group in groups.values())

    def _eligible_candidates(self, population, minimum_size):
        if self.composition == "constant":
            return population.candidates
        weights = np.array([1.0 / len(group)**0.5 if len(group) >= minimum_size else 0.0
                            for group in self.groups])
        if weights.sum() == 0:
            return ()
        indices = list(range(len(self.groups)))
        index = self.rng.choice(indices, size=1, replace=False, p=weights / weights.sum())[0]
        return self.groups[index]

    def _select(self, population, size, with_history):
        candidates = self._eligible_candidates(population, size)
        if len(candidates) < size:
            return None
        fitness = compute_population_fitness(candidates)
        if with_history:
            # Preserve the existing order of floating-point operations and RNG
            # draws so restart streams and seeded selections remain unchanged.
            fitness /= np.sqrt([1 + self.participation.get(a.info["confid"], 0) for a in candidates])
            fitness /= np.sqrt([1 + population.similarity_counts.get(a.info["confid"], 0) for a in candidates])
        indices = self.rng.choice(len(candidates), size=size, replace=False, p=fitness / fitness.sum())
        return [candidates[i] for i in indices]

    def select_one(self, population, with_history=True):
        selected = self._select(population, 1, with_history)
        return selected[0] if selected is not None else None

    def select_pair(self, population, with_history=True):
        selected = self._select(population, 2, with_history)
        return tuple(selected) if selected is not None else None
