"""Adaptive exploration algorithms built on the execution-service boundary."""

from gdpx.core.registry import Registry

from .protocol import Exploration, ExplorationResult, ExplorationStrategy, Proposal

REGISTER = Registry("exploration")

REGISTER.register_lazy("global_optimisation", "gdpx.exploration.population.exploration", "create_global_optimisation")
REGISTER.register_lazy("hybrid_monte_carlo", "gdpx.exploration.monte_carlo.hybrid_monte_carlo", "HybridMonteCarlo")
REGISTER.register_lazy("monte_carlo", "gdpx.exploration.monte_carlo.monte_carlo", "create_monte_carlo")
REGISTER.register_lazy("simulated_annealing", "gdpx.exploration.simulated_annealing.simulated_annealing", "SimulatedAnnealing")
REGISTER.register_lazy("artificial_reaction", "gdpx.exploration.artificial_force.afir", "AFIRSearch")

__all__ = ["Exploration", "ExplorationResult", "ExplorationStrategy", "Proposal", "REGISTER"]
