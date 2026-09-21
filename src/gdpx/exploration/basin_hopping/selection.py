"""Fitness-weighted selection of independent hopping-chain starts."""
from ..population.population import compute_population_fitness


class HoppingStartSelector:
    def __init__(self, rng, replace=False):
        if not isinstance(replace, bool):
            raise TypeError("BH selection.replace must be a boolean.")
        self.rng = rng
        self.replace = replace

    def select(self, population, count, with_history=True):
        candidates = population.candidates
        if not candidates:
            return []
        fitness = compute_population_fitness(
            candidates, population.similarity_counts if with_history else None
        )
        indices = self.rng.choice(len(candidates), size=count, replace=self.replace or len(candidates) < count,
                                  p=fitness / fitness.sum())
        return [candidates[i] for i in indices]
