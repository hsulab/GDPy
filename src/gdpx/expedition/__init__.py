#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import traceback

from .. import config
from ..core.register import registers
from ..builder.builder import StructureBuilder
from ..worker.single import SingleWorker
from ..worker.drive import DriverBasedWorker
from ..worker.interface import ComputerVariable
from ..utils.command import convert_indices

from .interface import ExpeditionVariable
registers.variable.register(ExpeditionVariable)

from .interface import explore 
registers.operation.register(explore)

from .expedition import AbstractExpedition

from .bh.bh import BasinHopping
registers.expedition.register("basin_hopping")(BasinHopping)

from .ga.engine import GeneticAlgorithemEngine
registers.expedition.register("genetic_algorithm")(GeneticAlgorithemEngine)

from .mc.mc import MonteCarlo
registers.expedition.register("monte_carlo")(MonteCarlo)

from .simulated_annealing.simulated_annealing import SimulatedAnnealing
registers.expedition.register("simulated_annealing")(SimulatedAnnealing)

# - optional
try:
    from .af.afir import AFIRSearch
    registers.expedition.register("artificial_reaction")(AFIRSearch)
except:
    config._print("AFIR is not loaded.")
    config._print(traceback.print_exc())


if __name__ == "__main__":
    ...