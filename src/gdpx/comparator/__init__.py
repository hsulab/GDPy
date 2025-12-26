from gdpx.core.register import BaseRegister

REGISTER = BaseRegister("comparator")

from .cartesian import CartesianComparator

REGISTER.register("cartesian")(CartesianComparator)

from .coordination import CoordinationComparator

REGISTER.register("coordination")(CoordinationComparator)

from .graph import GraphComparator

REGISTER.register("graph")(GraphComparator)

from .singlepoint import SinglePointComparator

REGISTER.register("single_point")(SinglePointComparator)

from .reaction import ReactionComparator

REGISTER.register("reaction")(ReactionComparator)
