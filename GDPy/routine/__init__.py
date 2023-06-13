#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from ..core.register import registers
from .ga.engine import GeneticAlgorithemEngine, GeneticAlgorithmVariable
from .mc.mc import MonteCarlo, MonteCarloVariable

registers.routine.register("genetic_algorithm")(GeneticAlgorithemEngine)
registers.routine.register("monte_carlo")(MonteCarlo)

registers.variable.register("GeneticAlgorithmVariable")(MonteCarloVariable)
registers.variable.register("MonteCarloVariable")(MonteCarloVariable)

from .interface import routine
registers.operation.register(routine)

if __name__ == "__main__":
    ...