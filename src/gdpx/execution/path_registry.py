#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from gdpx.core.register import Register

""" This submodule is for exploring, sampling, 
    and performing (chemical) reactions with
    various advanced algorithms.
"""

# Driver (reactor) backends
register_reactors = Register("reactor")

# String methods
from gdpx.providers.ase.path import AseStringReactor
from gdpx.providers.cp2k.path import Cp2kStringReactor
from gdpx.providers.grid.path import ZeroStringReactor
from gdpx.providers.vasp.path import VaspStringReactor

register_reactors.register("ase")(AseStringReactor)
register_reactors.register("cp2k")(Cp2kStringReactor)
register_reactors.register("vasp")(VaspStringReactor)
register_reactors.register("grid")(ZeroStringReactor)


if __name__ == "__main__":
    ...
