#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import inspect

from ase.ga.ofp_comparator import OFPComparator
from ase.ga.particle_comparator import NNMatComparator
from ase.ga.standard_comparators import InteratomicDistanceComparator

from GDPy.core.register import registers

registers.comparator.register(OFPComparator)
registers.comparator.register(NNMatComparator)
registers.comparator.register(InteratomicDistanceComparator)

@registers.comparator.register
class GraphComparator:

    def __init__(self) -> None:
        ...

#print(registers.comparator._dict)
#comp = registers.get("comparator", "interatomic_distance")
#print(comp)
#args = inspect.getargspec(comp.__init__).args
#print(args)

if __name__ == "__main__":
    ...