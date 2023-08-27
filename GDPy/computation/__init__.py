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

from GDPy.computation.cp2k import Cp2kDriver
register_drivers.register("cp2k")(Cp2kDriver)


if __name__ == "__main__":
    ...