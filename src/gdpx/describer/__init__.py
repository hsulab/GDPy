#!/usr/bin/env python3
# -*- coding: utf-8 -*-


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

from .formation_energy import FormationEnergyDescriber

REGISTER.register("formation_energy")(FormationEnergyDescriber)


if __name__ == "__main__":
    ...
