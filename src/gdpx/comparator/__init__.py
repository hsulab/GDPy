#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from ase.ga.ofp_comparator import OFPComparator
from ase.ga.particle_comparator import NNMatComparator

from ..core.register import registers

registers.comparator.register("OfpComparator")(OFPComparator)
registers.comparator.register(NNMatComparator)

from .inter_atomic_distance import InteratomicDistanceComparator
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
