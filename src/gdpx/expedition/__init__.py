#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from gdpx.group.group import create_a_group, create_a_molecule_group

from .. import config
from ..builder.builder import StructureBuilder
from ..utils.atoms_tags import get_tags_per_species
from ..core.register import registers
from ..data.array import AtomsNDArray
from ..graph.molecule import MolecularAdsorbate, find_molecules
from ..potential.interface import create_mixer
from ..utils.command import convert_indices, dict2str
from ..utils.strconv import str2array
from ..worker.drive import DriverBasedWorker
from ..worker.grid import GridDriverBasedWorker
from ..worker.interface import ComputerVariable
from ..worker.single import SingleWorker

from ..geometry.composition import convert_string_to_atoms
from ..geometry.bounce import bounce_one_atom


if __name__ == "__main__":
    ...
