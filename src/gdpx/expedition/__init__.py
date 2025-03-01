#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from gdpx.core.register import BaseRegister

REGISTER = BaseRegister("expedition")

# Evolutionary Methods
from .genetic_algorithm.engine import GeneticAlgorithmBroadcaster

REGISTER.register("genetic_algorithm")(GeneticAlgorithmBroadcaster)

# Monte Carlo Based Methods
from .monte_carlo.basin_hopping import BasinHopping

REGISTER.register("basin_hopping")(BasinHopping)

from .monte_carlo.hybrid_monte_carlo import HybridMonteCarlo

REGISTER.register("hybrid_monte_carlo")(HybridMonteCarlo)

from .monte_carlo.concurrent_hopping import ConcurrentHopping

REGISTER.register("concurrent_hopping")(ConcurrentHopping)

from .monte_carlo.monte_carlo import MonteCarlo

REGISTER.register("monte_carlo")(MonteCarlo)

# Other Methods
from .simulated_annealing.simulated_annealing import SimulatedAnnealing

REGISTER.register("simulated_annealing")(SimulatedAnnealing)

from .artificial_force.afir import AFIRSearch

REGISTER.register("artificial_reaction")(AFIRSearch)


if __name__ == "__main__":
    ...
