#!/usr/bin/env python3
# -*- coding: utf-8 -*


from .. import config
from ..core.register import Register

# Driver (dynamics) backends
register_drivers = Register("driver")

from gdpx.computation.asedriver import AseDriver
register_drivers.register("ase")(AseDriver)

try:
    from .jarex import JarexDriver
    register_drivers.register("jax")(JarexDriver)
except ImportError as e:
    config._print(f"  {'Driver Backend':<16s} {'`jax`':<16s} -> require `{e.name}`.")

try:
    from .deepmd_jax import DeepmdJaxDriver
    register_drivers.register("deepmd_jax")(DeepmdJaxDriver)
except ImportError as e:
    config._print(f"  {'Driver Backend':<16s} {'`deepmd_jax`':<16s} -> require `{e.name}`.")

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

from gdpx.computation.replica import ReplicaDriver
register_drivers.register("replica")(ReplicaDriver)


if __name__ == "__main__":
    ...
