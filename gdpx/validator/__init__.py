#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import warnings
import pathlib
from pathlib import Path
from typing import NoReturn, List, Union

from ..core.register import registers

from .dimer import DimerValidator
registers.validator.register("dimer")(DimerValidator)

from .mdf import MassDistributionValidator
registers.validator.register("mass_distribution")(MassDistributionValidator)

from .rdf import RdfValidator
registers.validator.register("radial_distribution")(RdfValidator)

from .eos import EquationOfStateValidator
registers.validator.register("equation_of_state")(EquationOfStateValidator)

from .melting_point import MeltingPointValidator
registers.validator.register("melting_point")(MeltingPointValidator)

from .minima import MinimaValidator
registers.validator.register("minima")(MinimaValidator)

from .spc import SinglepointValidator
registers.validator.register("spc")(SinglepointValidator)

from .surface_energy import SurfaceEnergyValidator
registers.validator.register("surface_energy")(SurfaceEnergyValidator)

from .diffusion_coefficient import DiffusionCoefficientValidator
registers.validator.register("diffusion_coefficient")(DiffusionCoefficientValidator)

from .rank import RankValidator
registers.validator.register("rank")(RankValidator)

from .rxn import PathwayValidator
registers.validator.register("mep")(PathwayValidator)


"""
Various properties to be validated

Atomic Energy and Crystal Lattice constant

Elastic Constants

Phonon Calculations

Point Defects (vacancies, self interstitials, ...)

Surface energies

Diffusion Coefficient

Adsorption, Reaction, ...
"""


def run_validation(params: dict, directory: Union[str, pathlib.Path], potter):
    """ This is a factory to deal with various validations...
    """
    # run over validations
    directory = pathlib.Path(directory)

    raise NotImplementedError("Command Line Validation is NOT Suppoted.")


if __name__ == "__main__":
    ...