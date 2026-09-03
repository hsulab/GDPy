#!/usr/bin/env python3
# -*- coding: utf-8 -*


from .. import config
from ..core.register import Register

# Driver (dynamics) backends
register_drivers = Register("driver")

from gdpx.providers.ase.driver import AseDriver
register_drivers.register("ase")(AseDriver)

try:
    from gdpx.providers.jax.driver import JarexDriver
    register_drivers.register("jax")(JarexDriver)
except ImportError as e:
    config._print(f"  {'Driver Backend':<16s} {'`jax`':<16s} -> require `{e.name}`.")

try:
    from gdpx.providers.deepmd.jax_driver import DeepmdJaxDriver
    register_drivers.register("deepmd_jax")(DeepmdJaxDriver)
except ImportError as e:
    config._print(f"  {'Driver Backend':<16s} {'`deepmd_jax`':<16s} -> require `{e.name}`.")

from gdpx.providers.lammps.execution import LmpDriver
register_drivers.register("lammps")(LmpDriver)

from gdpx.providers.lasp.driver import LaspDriver
register_drivers.register("lasp")(LaspDriver)

from gdpx.providers.abacus.driver import AbacusDriver
register_drivers.register("abacus")(AbacusDriver)

from gdpx.providers.vasp.driver import VaspDriver
register_drivers.register("vasp")(VaspDriver)

from gdpx.providers.cp2k.driver import Cp2kDriver
register_drivers.register("cp2k")(Cp2kDriver)

from gdpx.providers.replica.driver import ReplicaDriver
register_drivers.register("replica")(ReplicaDriver)


if __name__ == "__main__":
    ...
