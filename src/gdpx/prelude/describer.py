#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from gdpx.core.register import registers

from gdpx.describer.spc import SpcDescriber
registers.describer.register("spc")(SpcDescriber)

from gdpx.describer.coordinate import CoordinateDescriber
registers.describer.register("coordinate")(CoordinateDescriber)

from gdpx.describer.coordination import CoordinationDescriber
registers.describer.register("coordination")(CoordinationDescriber)

from gdpx.describer.connectivity import ConnectivityDescriber
registers.describer.register("connectivity")(ConnectivityDescriber)

from gdpx.describer.formation_energy import FormationEnergyDescriber
registers.describer.register("formation_energy")(FormationEnergyDescriber)


if __name__ == "__main__":
    ...
  
