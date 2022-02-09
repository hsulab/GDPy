#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

from ase.io import read
from ase.constraints import FixAtoms
from ase.calculators.emt import EMT
from ase.neb import NEB
#from ase.neb import SingleCalculatorNEB
from ase.optimize import BFGS
from ase.io import read, write

from eann.interface.ase.calculator import Eann

#initial = read('initial.traj')
#final = read('final.traj')

#prepared_images = read(
#    "/users/40247882/scratch2/validations/structures/surfaces-2x2/relaxed/hcp2fcc-brg_opt.xyz", ":" # read neb optimised trajectory
#)
#
#for atoms in prepared_images:
#    atoms.wrap()
#write("wrapped_traj.xyz", prepared_images)
#exit()

#prepared_images = read("./start_images.xyz", ":")
prepared_images = read("./hcp2fcc-top_opt.xyz", ":")

initial = prepared_images[0].copy()
final = prepared_images[-1].copy()

constraint = FixAtoms(indices=[1,2,3,4]) # first atom is O
#constraint = FixAtoms(indices=[1,2,3,4,5,6,7,8]) # first atom is O

nimages = 7
images = [initial]
images += [initial.copy() for i in range(nimages-2)]
images.append(final)

# set calculator
cur_model = "/users/40247882/scratch2/oxides/eann-main/reduce-13/ensemble/model-0/eann_best_DOUBLE.pt"
calc = Eann(
    type_map = {'O': 0, 'Pt': 1},
    model = cur_model
)

for atoms in images:
    calc = Eann(
        type_map = {'O': 0, 'Pt': 1},
        #model = "/users/40247882/projects/oxides/eann-main/it-0012/ensemble/model-3/eann_best-3_DOUBLE.pt"
        model = cur_model
    )
    atoms.calc = calc
    atoms.set_constraint(constraint)

print(initial.get_potential_energy())
#print(initial.get_forces())
print(final.get_potential_energy())

neb = NEB(
    images, allow_shared_calculator=False,
    k=0.1
    # dynamic_relaxation = False
)
#neb = SingleCalculatorNEB(images)

neb.interpolate()
#print(neb.images)

qn = BFGS(neb, trajectory="neb.traj")
qn.run(fmax=0.05, steps=50)

opt_images = read("./neb.traj", "-%s:" %nimages)
for a in opt_images:
    calc = Eann(
        type_map = {'O': 0, 'Pt': 1},
        model = cur_model
    )
    a.calc = calc

energies = np.array(
    [a.get_potential_energy() for a in opt_images]
)
energies = energies - energies[0]
print(energies)

if __name__ == '__main__':
    pass
