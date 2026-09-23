from gdpx.core.register import BaseRegister

REGISTER = BaseRegister("comparator")

from .cartesian import CartesianCoordinateComparator

REGISTER.register("cartesian_coordinate")(CartesianCoordinateComparator)

from .coordination import CoordinationComparator

REGISTER.register("coordination")(CoordinationComparator)

from .graph import GraphComparator

REGISTER.register("graph")(GraphComparator)

from .singlepoint import SinglePointComparator

REGISTER.register("single_point")(SinglePointComparator)

from .reaction import ReactionComparator

REGISTER.register("reaction")(ReactionComparator)
