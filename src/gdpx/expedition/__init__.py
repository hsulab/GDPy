#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import traceback

from .. import config
from ..core.register import registers
from ..builder.builder import StructureBuilder
from ..builder.utils import convert_string_to_atoms
from ..worker.single import SingleWorker
from ..worker.drive import DriverBasedWorker
from ..worker.interface import ComputerVariable
from ..utils.command import convert_indices, dict2str

from .interface import ExpeditionVariable
registers.variable.register(ExpeditionVariable)

from .interface import explore 
registers.operation.register(explore)

from .expedition import AbstractExpedition

from .ga.engine import GeneticAlgorithemEngine
registers.expedition.register("genetic_algorithm")(GeneticAlgorithemEngine)

from .monte_carlo.basin_hopping import BasinHopping
registers.expedition.register("basin_hopping")(BasinHopping)

from .monte_carlo.monte_carlo import MonteCarlo
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