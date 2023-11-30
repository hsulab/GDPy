#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from ase.ga.particle_crossovers import CutSpliceCrossover
from ase.ga.cutandsplicepairing import CutAndSplicePairing

from gdpx.core.register import registers

registers.builder.register("cut_and_splice")(CutAndSplicePairing)
registers.builder.register("cut_and_splice_cluster")(CutSpliceCrossover)



if __name__ == "__main__":
    ...