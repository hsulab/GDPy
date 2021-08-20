#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import argparse
from ase.geometry.geometry import get_angles_derivatives

from ase.io import read, write
from ase.constraints import FixAtoms

from GDPy.mc.gcmc import ReducedRegion, GCMC

"""
constraints are read from xyz property move_mask
"""

parser = argparse.ArgumentParser()
parser.add_argument(
    'INPUT', 
    default='gc.json', 
    help='grand canonical inputs'
)
parser.add_argument(
    '-r', '--run', action='store_true',
    help='run GA procedure'
)


args = parser.parse_args()

with open(args.INPUT, 'r') as fopen:
    gc_dict = json.load(fopen)

# set initial structure - bare metal surface
atoms = read('/users/40247882/projects/oxides/gdp-main/mc-test/Pt_111_0.xyz')

#res = Reservior(particle='O', temperature=300, pressure=1.0, mu=chemical_potential) # Kelvin, atm, eV

# set region
region = ReducedRegion(atoms.cell, caxis=[4.5,12.5], mindis=1.5)

# start mc
type_map = {'O': 0, 'Pt': 1}
transition_array = [0.5,0.5] # move and exchange
gcmc = GCMC(type_map, gc_dict["reservior"], atoms, region, 5, transition_array)

# set calculator

#run_params = RunParam(
#    backend='reax', calculator=calc, optimiser='lammps', constraint=cons,
#    convergence = [0.05, 200]
#)

if args.run:
    gcmc.run(gc_dict["calculation"])


if __name__ == "__main__":
    pass