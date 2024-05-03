#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from ..core.register import registers
from ..utils.strconv import str2array

"""Add bias on potential energy surface. 

Some of bias forces are based on JAX. In the future, we need replace those oft-used
ones to pure python codes as jax need accelerate them a lot.

"""

from .afir import AFIRCalculator

registers.bias.register("afir")(AFIRCalculator)

from .bondboost import BondBoostCalculator

registers.bias.register("bondboost")(BondBoostCalculator)

from .nuclei import NucleiRepulsionCalculator

registers.bias.register("nuclei_repulsion")(NucleiRepulsionCalculator)

from .harmonic import DistanceHarmonicCalculator, PlaneHarmonicCalculator

registers.bias.register("distance_harmonic")(DistanceHarmonicCalculator)
registers.bias.register("plane_harmonic")(PlaneHarmonicCalculator)

from .gaussian import (BondGaussianCalculator, CenterOfMassGaussianCalculator,
                       DistanceGaussianCalculator, RMSDGaussian)

registers.bias.register("bond_gaussian")(BondGaussianCalculator)
registers.bias.register("center_of_mass_gaussian")(CenterOfMassGaussianCalculator)
registers.bias.register("distance_gaussian")(DistanceGaussianCalculator)
registers.bias.register("rmsd_gaussian")(RMSDGaussian)


if __name__ == "__main__":
    ...
