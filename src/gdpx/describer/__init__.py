from gdpx import config
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

try:
    from .soap import SoapDescriber

    REGISTER.register("soap")(SoapDescriber)
except ImportError as err:
    config._print(f"  {'Describer':<16s} {'`soap`':<16s} -> require `{err.name}`.")
