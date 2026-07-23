from gdpx.core.register import BaseRegister

REGISTER = BaseRegister("comparator")

REGISTER.register_lazy("cartesian_coordinate", "gdpx.comparator.cartesian", "CartesianCoordinateComparator")
REGISTER.register_lazy("coordination", "gdpx.comparator.coordination", "CoordinationComparator")
REGISTER.register_lazy("graph", "gdpx.comparator.graph", "GraphComparator")
REGISTER.register_lazy("single_point", "gdpx.comparator.singlepoint", "SinglePointComparator")
REGISTER.register_lazy("reaction", "gdpx.comparator.reaction", "ReactionComparator")
