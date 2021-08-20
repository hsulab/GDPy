#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

from ase.ga.data import DataConnection
from ase.io import read, write

#da = DataConnection('/users/40247882/scratch2/voxaop/surfaces-1/goffee-ga/vgadb.db')
da = DataConnection('/users/40247882/scratch2/voxaop/surfaces-1/O12Al9V1/vgadb2.db')
print('number of unrelaxed structures', da.get_number_of_unrelaxed_candidates())
print(da.__get_ids_of_all_unrelaxed_candidates__())

#print('check populationsize')
row = da.c.get(1)
#new_data = row['data'].copy()
#new_data['population_size'] = 20
#da.c.update(1, data=new_data)
print(row['data'])
exit()

#for row in da.c.select(pairing=1,generation=1):
#    print(row['id'])
#    print(row['generation'])
#    print(row['key_value_pairs'])
#    rowid = row['id']
#    #rowdict = row['key_value_pairs']
#    #rowdict['generation'] = 1
#    da.c.update(rowid, generation=1)

# check queued
#for idx, row in enumerate(da.c.select('queued=1')):
#    print(idx, ' ', row['id'])
#    print(row['key_value_pairs'])

#for idx, row in enumerate(da.c.select('pairing=1')):
#    print(idx, ' ', row['id'])
#    #print(row['key_value_pairs'])
#    print(row['data'])
#
#exit()

# get unrelaxed gaids
gaids = []
for idx, row in enumerate(da.c.select('relaxed=1,generation=1')):
    print(idx, ' ', row['id'], ' ', row['gaid'])
    queued = row.get('queued', None)
    #if queued is None:
    #    print(row['key_value_pairs'])
    #     gaids.append(row['gaid'])
    #print(row['key_value_pairs'])
    #if row['gaid'] > 108:
    #    da.c.update(row['id'], generation=2)


exit()

#da.c.update(123, generation=1)
for idx, row in enumerate(da.c.select('relaxed=1,generation=0')):
    print(idx, '  ', row['id'])
    #origin = row.get('origin', None)
    #if origin is not None:
    #    if origin == 'CutAndSplicePairing':
    #        # da.c.update(row['id'], generation=1)
    #        print(row['key_value_pairs'])
    if row['gaid'] not in gaids:
        print(row['key_value_pairs'])
        #da.c.update(row['id'], generation=1)

#row = da.c.get(selection=101)
#print(row['key_value_pairs'])
#atoms = da.c.get_atoms(selection=101)
#print(atoms.info)
#row = da.c.get(selection=2)
#confs = da.get_all_candidates_in_queue()
#print(confs)

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

# remove queued jobs
for confid in range(11,22):
    print('confid ', confid)
    da.remove_from_queue(confid)
for confid in da.get_all_candidates_in_queue():
    print('confid ', confid)
    #da.remove_from_queue(confid)
"""
