#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy

from ..core.register import registers


""" This submodule is for exploring, sampling, 
    and performing (chemical) reactions with
    various advanced algorithms.
"""

# - string methods...
from .grid import ZeroStringReactor
registers.reactor.register("grid")(ZeroStringReactor)

from .pathway import MEPFinder
registers.reactor.register("ase")(MEPFinder)

from .cp2k import Cp2kStringReactor
registers.reactor.register("cp2k")(Cp2kStringReactor)

from .vasp import VaspStringReactor
registers.reactor.register("vasp")(VaspStringReactor)


if __name__ == "__main__":
    ...