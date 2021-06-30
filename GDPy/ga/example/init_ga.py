#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from ase.ga.data import PrepareDB
from ase.ga.startgenerator import StartGenerator
from ase.ga.utilities import closest_distances_generator
from ase.ga.utilities import get_all_atom_types
from ase.constraints import FixAtoms
import numpy as np
from ase.build import fcc111

db_file = 'gadb.db'

# create the surface
slab = fcc111('Au', size=(4, 4, 1), vacuum=10.0, orthogonal=True)
slab.set_constraint(FixAtoms(mask=len(slab) * [True]))

# define the volume in which the adsorbed cluster is optimized
# the volume is defined by a corner position (p0)
# and three spanning vectors (v1, v2, v3)
pos = slab.get_positions()
cell = slab.get_cell()
p0 = np.array([0., 0., max(pos[:, 2]) + 2.])
v1 = cell[0, :] * 0.8
v2 = cell[1, :] * 0.8
v3 = cell[2, :]
v3[2] = 3.

# Define the composition of the atoms to optimize
atom_numbers = 2 * [47] + 2 * [79]

# define the closest distance two atoms of a given species can be to each other
unique_atom_types = get_all_atom_types(slab, atom_numbers)
blmin = closest_distances_generator(atom_numbers=unique_atom_types,
                                    ratio_of_covalent_radii=0.7)

# create the starting population
sg = StartGenerator(slab, atom_numbers, blmin,
                    box_to_place_in=[p0, [v1, v2, v3]])

# generate the starting population
population_size = 20
starting_population = [sg.get_new_candidate() for i in range(population_size)]

# from ase.visualize import view   # uncomment these lines
# view(starting_population)        # to see the starting population

# create the database to store information in
d = PrepareDB(db_file_name=db_file,
              simulation_cell=slab,
              stoichiometry=atom_numbers)

for a in starting_population:
    d.add_unrelaxed_candidate(a)