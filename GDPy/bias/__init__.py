#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from typing import List

from ..core.register import Register
bias_register = Register("bias")

from ..colvar import initiate_colvar

from .afir import AFIRCalculator
bias_register.register("afir")(AFIRCalculator)

from .harmonic import HarmonicBias
bias_register.register("harmonic")(HarmonicBias)


"""Create bias on potential energy surface. These are all JAX-based PES modifiers.
"""


if __name__ == "__main__":
    ...