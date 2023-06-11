#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from ase.ga.ofp_comparator import OFPComparator
from ase.ga.particle_comparator import NNMatComparator
from ase.ga.standard_comparators import InteratomicDistanceComparator

from ..core.register import registers
from .graph import GraphComparator

registers.comparator.register(OFPComparator)
registers.comparator.register(NNMatComparator)
registers.comparator.register(InteratomicDistanceComparator)

registers.comparator.register(GraphComparator)


if __name__ == "__main__":
    ...