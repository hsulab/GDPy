#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test DP calculator with multiple models
"""

# read structures 
from ase.io import read, write
atoms = read('../templates/structures/Pt_opt.xyz')

# set calculator
type_map = {'O': 0, 'Pt': 1}
model = [
    '/users/40247882/projects/oxides/gdp-main/it-0009/ensemble/model-0/graph.pb', 
    '/users/40247882/projects/oxides/gdp-main/it-0009/ensemble/model-1/graph.pb', 
    '/users/40247882/projects/oxides/gdp-main/it-0009/ensemble/model-2/graph.pb', 
    '/users/40247882/projects/oxides/gdp-main/it-0009/ensemble/model-3/graph.pb'
]


from GDPy.calculator.dp import DP
calc = DP(model=model, type_dict=type_map, njobs=4)
atoms.calc = calc

from ase import units
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from GDPy.md.md_utils import force_temperature
from GDPy.md.nosehoover import NoseHoover

# set MD
temperature = 300
MaxwellBoltzmannDistribution(atoms, temperature*units.kB)
force_temperature(atoms, temperature)

timestep = 2.0

nvt_dyn = NoseHoover(
    atoms = atoms,
    timestep = timestep * units.fs,
    temperature = temperature * units.kB,
    nvt_q = 334.
)

# run MD
import time
st = time.time()
nsteps = 100
for n in range(nsteps):
    print('step ', n)
    nvt_dyn.step()
et = time.time()
print('cost tims: ', et - st)