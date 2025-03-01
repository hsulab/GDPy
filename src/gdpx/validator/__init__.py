#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from gdpx.core.register import BaseRegister

REGISTER = BaseRegister("validator")

from .dimer import DimerValidator

REGISTER.register("dimer")(DimerValidator)

from .trimer import TrimerValidator

REGISTER.register("trimer")(TrimerValidator)

from .mass_distribution import MassDistributionValidator

REGISTER.register("mass_distribution")(MassDistributionValidator)

from .radial_distribution import RdfValidator

REGISTER.register("radial_distribution")(RdfValidator)

from .equation_of_state import EquationOfStateValidator

REGISTER.register("equation_of_state")(EquationOfStateValidator)

from .melting_point import MeltingPointValidator

REGISTER.register("melting_point")(MeltingPointValidator)

from .minima import MinimaValidator

REGISTER.register("minima")(MinimaValidator)

from .spc import SinglepointValidator

REGISTER.register("spc")(SinglepointValidator)

from .surface_energy import SurfaceEnergyValidator

REGISTER.register("surface_energy")(SurfaceEnergyValidator)

from .mean_squared_displacement import MeanSquaredDisplacementValidator

REGISTER.register("mean_squared_displacement")(MeanSquaredDisplacementValidator)
REGISTER.register("diffusion_coefficient")(MeanSquaredDisplacementValidator)

from .rank import RankValidator

REGISTER.register("rank")(RankValidator)

from .rxn import PathwayValidator

REGISTER.register("mep")(PathwayValidator)


if __name__ == "__main__":
    ...
