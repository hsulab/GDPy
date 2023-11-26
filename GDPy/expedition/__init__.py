#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from ..core.register import registers
from ..core.variable import Variable
from ..worker.single import SingleWorker

from .bh.bh import BasinHopping, BasinHoppingVariable
from .ga.engine import GeneticAlgorithemEngine, GeneticAlgorithmVariable
from .mc.mc import MonteCarlo, MonteCarloVariable

registers.expedition.register("basin_hopping")(BasinHopping)
registers.expedition.register("genetic_algorithm")(GeneticAlgorithemEngine)
registers.expedition.register("monte_carlo")(MonteCarlo)

registers.variable.register("BasinHoppingVariable")(BasinHoppingVariable)
registers.variable.register("GeneticAlgorithmVariable")(GeneticAlgorithmVariable)
registers.variable.register("MonteCarloVariable")(MonteCarloVariable)

from .interface import explore 
registers.operation.register(explore)

if __name__ == "__main__":
    ...