"""Fitness-weighted selection of independent hopping-chain starts."""
from ..population.population import compute_population_fitness


class HoppingStartSelector:
    def __init__(self, rng):
        self.rng = rng

    def select(self, population, count, with_history=True):
        candidates = population.candidates
        if not candidates:
            return []
        fitness = compute_population_fitness(
            candidates, population.similarity_counts if with_history else None
        )
        indices = self.rng.choice(len(candidates), size=count, replace=True, p=fitness / fitness.sum())
        return [candidates[i] for i in indices]
