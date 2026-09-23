#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from gdpx.core.register import BaseRegister

"""Add bias on potential energy surface. 

Some of bias forces are based on JAX. In the future, we need replace those oft-used
ones to pure python codes as jax need accelerate them a lot.

"""

REGISTER = BaseRegister("bias")

from .afir import AFIRCalculator

REGISTER.register("afir")(AFIRCalculator)

from .bondboost import BondBoostCalculator

REGISTER.register("bondboost")(BondBoostCalculator)

from .nuclei import NucleiRepulsionCalculator

REGISTER.register("nuclei_repulsion")(NucleiRepulsionCalculator)

from .harmonic import DistanceHarmonicCalculator, PlaneHarmonicCalculator

REGISTER.register("distance_harmonic")(DistanceHarmonicCalculator)
REGISTER.register("plane_harmonic")(PlaneHarmonicCalculator)

from .gaussian import BondGaussianCalculator, CenterOfMassGaussianCalculator, DistanceGaussianCalculator, RMSDGaussian

REGISTER.register("bond_gaussian")(BondGaussianCalculator)
REGISTER.register("center_of_mass_gaussian")(CenterOfMassGaussianCalculator)
REGISTER.register("distance_gaussian")(DistanceGaussianCalculator)
REGISTER.register("rmsd_gaussian")(RMSDGaussian)


if __name__ == "__main__":
    ...
