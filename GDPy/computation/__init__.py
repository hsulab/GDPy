#!/usr/bin/env python3
# -*- coding: utf-8 -*

from ..core.register import Register

# - driver (dynamics) backends...
register_drivers = Register("driver")

from GDPy.computation.asedriver import AseDriver
register_drivers.register("ase")(AseDriver)

from GDPy.computation.lammps import LmpDriver
register_drivers.register("lammps")(LmpDriver)

from GDPy.computation.lasp import LaspDriver
register_drivers.register("lasp")(LaspDriver)

from GDPy.computation.vasp import VaspDriver
register_drivers.register("vasp")(VaspDriver)


if __name__ == "__main__":
    ...