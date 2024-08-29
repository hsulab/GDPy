#!/usr/bin/env python3
# -*- coding: utf-8 -*

from .. import config
from ..core.register import Register

# - driver (dynamics) backends...
register_drivers = Register("driver")

from gdpx.computation.asedriver import AseDriver
register_drivers.register("ase")(AseDriver)

try:
    from .jarex import JarexDriver
    register_drivers.register("jax")(JarexDriver)
except ImportError:
    config._print(f"Driver Backend `jax` is not imported.")

from gdpx.computation.lammps import LmpDriver
register_drivers.register("lammps")(LmpDriver)

from gdpx.computation.lasp import LaspDriver
register_drivers.register("lasp")(LaspDriver)

from .abacus import AbacusDriver
register_drivers.register("abacus")(AbacusDriver)

from gdpx.computation.vasp import VaspDriver
register_drivers.register("vasp")(VaspDriver)

from gdpx.computation.cp2k import Cp2kDriver
register_drivers.register("cp2k")(Cp2kDriver)


if __name__ == "__main__":
    ...
