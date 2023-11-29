#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import traceback

from .. import config
from ..core.register import registers
from ..core.variable import Variable
from ..builder.builder import StructureBuilder
from ..worker.interface import ComputerVariable
from ..worker.single import SingleWorker
from ..worker.drive import DriverBasedWorker

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

# - optional
try:
    from .af.afir import AFIRSearch, ArtificialReactionVariable
    registers.expedition.register("artificial_reaction")(AFIRSearch)
    registers.variable.register("ArtificialReactionVariable")(ArtificialReactionVariable)
except:
    config._print("AFIR is not loaded.")
    config._print(traceback.print_exc())


if __name__ == "__main__":
    ...