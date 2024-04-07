#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy

from .. import config
from ..core.register import registers

from ..builder.constraints import parse_constraint_info
from ..potential.calculators.mixer import EnhancedCalculator

from ..utils.strucopy import read_sort, resort_atoms_with_spc
from ..utils.cmdrun import run_ase_calculator


""" This submodule is for exploring, sampling, 
    and performing (chemical) reactions with
    various advanced algorithms.
"""

# - string methods...
from .string import (
    AseStringReactor, Cp2kStringReactor, VaspStringReactor,
    ZeroStringReactor
)
registers.reactor.register("ase")(AseStringReactor)
registers.reactor.register("cp2k")(Cp2kStringReactor)
registers.reactor.register("vasp")(VaspStringReactor)
registers.reactor.register("grid")(ZeroStringReactor)


if __name__ == "__main__":
    ...
