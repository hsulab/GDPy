#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import json
from pathlib import Path

import numpy as np

from ase.io import read, write

"""
2294774514
4241608611
1079676012
3692161385
"""

"""
keywords
    model
        type_map
        descriptor
        fitting_net
    learning_rate
    loss
    training
"""

def run_command(directory, command, comment=''):
    proc = subprocess.Popen(
        command, shell=True, cwd=directory, 
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        encoding = 'utf-8'
    )
    errorcode = proc.wait(timeout=120) # 10 seconds
    if errorcode:
        raise ValueError('Error in %s at %s.' %(comment, directory))
    
    return ''.join(proc.stdout.readlines())

def generate_random_seed():
    drng = np.random.default_rng() # default random generator
    seed = drng.integers(sys.maxsize) % (2**32)

    return seed

def frames2dataset(frames):
    """"""
    atomic_properties = ['numbers', 'positions', 'forces']
    calc_props = ['energy', 'forces']

    for atoms in frames:
        # remove extra properties in atoms
        cur_properties = list(atoms.arrays.keys())
        for prop in cur_properties:
            if prop not in atomic_properties:
                #atoms.arrays.pop(prop)
                del atoms.arrays[prop]
        # atoms info 
        # del atoms.info['feature_vector']
        # TODO: check if calculator exists 
        atoms.calc = None # ase copys xyz info to SinglePointCalculator?
        atoms.arrays['force'] = atoms.arrays['forces'].copy()
        del atoms.arrays['forces']

    write('dp_raw.xyz', frames)

    # 
    xyz_multi_systems = dpdata.MultiSystems.from_file(
        file_name='./dp_raw.xyz', 
        fmt='quip/gap/xyz'
    )
    print(xyz_multi_systems)
    xyz_multi_systems.to_deepmd_raw('./raw_data/')

    return 

def find_systems(raw_data):
    """"""
    # read systems
    systems = []
    for p in raw_data.glob('O*'):
        systems.append(p)
    #systems.sort()

    return systems

def read_dptrain_json(iter_directory, main_database, main_dict):
    """"""
    # parse params
    machine_json = main_dict['machines']['trainer']
    num_models = main_dict['training']['num_models']
    train_json = main_dict['training']['json']

    # prepare dataset
    xyz_files = []
    for p in main_database.glob('*.xyz'):
        xyz_files.append(p)
    
    frames = []
    for xyz_file in xyz_files:
        frames.extend(read(xyz_file, ':'))
    
    print(len(frames))

    # find systems
    data_path = Path('/users/40247882/projects/oxides/gdp-main/raw_data')
    systems = find_systems(data_path)

    # machine file
    from ..machine.machine import SlurmMachine
    slurm_machine = SlurmMachine(machine_json)
    #slurm_machine.write('/users/40247882/projects/oxides/dptrain/test.slurm')

    # read json
    with open(train_json, 'r') as fopen:
        params_dict = json.load(fopen)
    # change seed and system dirs
    ensemble_dir = iter_directory / 'ensemble'
    ensemble_dir.mkdir()
    for idx in range(num_models):
        #print(params_dict)
        model_dir = ensemble_dir / ('model-'+str(idx))
        model_dir.mkdir()
        seed = generate_random_seed()
        params_dict['model']['descriptor']['seed'] = int(seed)
        params_dict['model']['fitting_net']['seed'] = int(seed)

        params_dict['training']['systems'] = [str(s) for s in systems]

        with open(model_dir/'dp.json', 'w') as fopen:
            json.dump(params_dict, fopen, indent=4)

        # write machine 
        slurm_machine.machine_dict['job-name'] = 'model-'+str(idx)
        slurm_machine.write(model_dir/'dptrain.slurm')

        # submit job
    
    # check job status
    
    return

if __name__ == '__main__':
    pass