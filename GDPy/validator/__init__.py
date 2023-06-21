#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import warnings
import pathlib
from pathlib import Path
from typing import NoReturn, List, Union

import numpy as np

import matplotlib
matplotlib.use('Agg') #silent mode
import matplotlib.pyplot as plt
#plt.style.use("presentation")

from ase import Atoms
from ase.io import read, write

from GDPy.utils.command import parse_input_file

from GDPy.validator.validator import AbstractValidator

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