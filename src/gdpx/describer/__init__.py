from gdpx.core.register import BaseRegister

REGISTER = BaseRegister("describer")

from .spc import SpcDescriber

REGISTER.register("spc")(SpcDescriber)

from .coordinate import CoordinateDescriber

REGISTER.register("coordinate")(CoordinateDescriber)

from .coordination import CoordinationDescriber

REGISTER.register("coordination")(CoordinationDescriber)

from .connectivity import ConnectivityDescriber

REGISTER.register("connectivity")(ConnectivityDescriber)

from .dissociative import DissociativeDescriber

REGISTER.register("dissociative")(DissociativeDescriber)

from .formation_energy import FormationEnergyDescriber

REGISTER.register("formation_energy")(FormationEnergyDescriber)

REGISTER.register_lazy("soap", "gdpx.describer.soap", "SoapDescriber")

from .cluster import ClusterDescriber

REGISTER.register("cluster")(ClusterDescriber)

from .colvar import ColvarDescriber

REGISTER.register("colvar")(ColvarDescriber)
