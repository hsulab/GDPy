#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

from ase.ga.data import DataConnection
from ase.io import read, write

from ase.calculators.vasp.create_input import GenerateVaspInput

from GDPy.ga.make_all_vasp import create_vasp_inputs

# pseudo 
pp_path = "/mnt/scratch/chemistry-apps/dkb01416/vasp/PseudoPotential"
if 'VASP_PP_PATH' in os.environ.keys():
    os.environ.pop('VASP_PP_PATH')
os.environ['VASP_PP_PATH'] = pp_path

# vdw 
vdw_envname = 'ASE_VASP_VDW'
vdw_path = "/mnt/scratch/chemistry-apps/dkb01416/vasp/pot"
if vdw_envname in os.environ.keys():
    _ = os.environ.pop(vdw_envname)
os.environ[vdw_envname] = vdw_path

da = DataConnection('/users/40247882/scratch2/voxaop/surfaces-1/goffee-ga/vgadb.db')
print(da.get_number_of_unrelaxed_candidates())

"""
for i in range(5):
    atoms = da.get_an_unrelaxed_candidate()
    print(atoms.info['confid'])

create_vasp = GenerateVaspInput()
for atoms in da.get_all_unrelaxed_candidates():
    print(atoms)
    print(atoms.info['confid'])
    break
    create_vasp.set_xc_params('PBE') # since incar not setting GGA
    create_vasp.read_incar('/users/40247882/scratch2/voxaop/surfaces-1/goffee-ga/INCAR')
    create_vasp.input_params['kpts'] = (2,3,1)
    create_vasp.input_params['gamma'] = True
    create_vasp.initialize(atoms)
    create_vasp.write_input(atoms, './vasp-test')
    #print(atoms.info['confid'])
    # create_vasp_inputs(atoms)
    #write('POSCAR', atoms, sort=True, vasp5=True)
    break
"""

# remove queued jobs
for confid in range(11,22):
    print('confid ', confid)
    da.remove_from_queue(confid)
for confid in da.get_all_candidates_in_queue():
    print('confid ', confid)
    #da.remove_from_queue(confid)