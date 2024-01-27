#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from ase.ga.ofp_comparator import OFPComparator
from ase.ga.particle_comparator import NNMatComparator
from ase.ga.standard_comparators import InteratomicDistanceComparator

from ..core.register import registers

registers.comparator.register(OFPComparator)
registers.comparator.register(NNMatComparator)
registers.comparator.register(InteratomicDistanceComparator)

from .cartesian import CartesianComparator
registers.comparator.register(CartesianComparator)

from .coordination import CoordinationComparator
registers.comparator.register(CoordinationComparator)

from .graph import GraphComparator
registers.comparator.register(GraphComparator)

from .singlepoint import SinglePointComparator
registers.comparator.register("single_point")(SinglePointComparator)

from .reaction import ReactionComparator
registers.comparator.register("reaction")(ReactionComparator)



if __name__ == "__main__":
    ...