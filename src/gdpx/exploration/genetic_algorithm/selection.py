"""Fitness-weighted GA selection with operator-defined parent compatibility."""
import numpy as np

from ..population.population import compute_population_fitness


class GeneticParentSelector:
    def __init__(self, rng):
        self.rng = rng
        self.participation = {}

    def refresh(self, population, database):
        participation, _ = database.get_participation_in_pairing()
        self.participation = dict(participation)

    def _weights(self, population, with_history):
        candidates = population.candidates
        fitness = compute_population_fitness(candidates)
        if with_history:
            fitness /= np.sqrt([1 + self.participation.get(a.info["confid"], 0) for a in candidates])
            fitness /= np.sqrt([1 + population.similarity_counts.get(a.info["confid"], 0) for a in candidates])
        return fitness

    def _choose(self, indices, weights):
        probabilities = weights[indices]
        return int(self.rng.choice(indices, p=probabilities / probabilities.sum()))

    def select_one(self, population, with_history=True):
        candidates = population.candidates
        if not candidates:
            return None
        index = self._choose(list(range(len(candidates))), self._weights(population, with_history))
        return candidates[index]

    def select_pair(self, population, with_history=True, *, compatible=None):
        # Compatibility filters membership; fitness is still measured against the full pool.
        candidates = population.candidates
        if len(candidates) < 2:
            return None
        partners = {
            i: [j for j in range(len(candidates)) if i != j and
                (compatible is None or compatible(candidates[i], candidates[j]))]
            for i in range(len(candidates))
        }
        eligible = [i for i, matches in partners.items() if matches]
        if not eligible:
            return None
        weights = self._weights(population, with_history)
        first = self._choose(eligible, weights)
        second = self._choose(partners[first], weights)
        return candidates[first], candidates[second]
