#!/usr/bin/env python3
# -*- coding: utf-8 -*-



from typing import List

from ..core.register import Register, registers
bias_register = Register("bias")

"""Add bias on potential energy surface. 

Some of bias forces are based on JAX. In the future, we need replace those oft-used
ones to pure python codes as jax need accelerate them a lot.

"""

from .afir import AFIRCalculator
bias_register.register("afir")(AFIRCalculator)

from .bondboost import BondBoostCalculator
bias_register.register("bondboost")(BondBoostCalculator)

# from .harmonic import HarmonicBias
# bias_register.register("harmonic")(HarmonicBias)

from .harmonic import PlaneHarmonicCalculator
bias_register.register("plane_harmonic")(PlaneHarmonicCalculator)

# from .gaussian import GaussianCalculator
# bias_register.register("gaussian")(GaussianCalculator)


if __name__ == "__main__":
    ...
