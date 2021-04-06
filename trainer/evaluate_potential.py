#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import subprocess
from pathlib import Path

import numpy as np

from ase.io import read, write

model_dirs = [
    'ensemble-1',
    'ensemble-2',
    'ensemble-3',
    'ensemble-4',
]

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

def check_convergence(model_path):
    """check if the training is finished"""
    converged = False
    model_path = Path(model_path)
    dpout_path = model_path / 'dp.out'
    if dpout_path.exists():
        content = dpout_path.read_text()
        line = content.split('\n')[-3]
        if 'finished' in line:
            converged = True
    
    return converged

def freeze_model(model_path):
    command = 'dp freeze -o graph.pb'
    output = run_command(model_path, command)
    print(output)
    return 

def freeze_ensemble():
    for model_dir in model_dirs:
        if check_convergence(model_dir):
            freeze_model(model_dir)

def rms_dict(x_ref, x_pred):
    """ Takes two datasets of the same shape and returns a dictionary containing RMS error data"""

    x_ref = np.array(x_ref)
    x_pred = np.array(x_pred)

    if np.shape(x_pred) != np.shape(x_ref):
        raise ValueError('WARNING: not matching shapes in rms')

    error_2 = (x_ref - x_pred) ** 2

    average = np.sqrt(np.average(error_2))
    std_ = np.sqrt(np.var(error_2))

    return {'rmse': average, 'std': std_}

def read_properties(frames):
    energies = [atoms.get_potential_energy() for atoms in frames]
    forces_dict = {}
    for atoms in frames:
        for sym, force in zip(atoms.get_chemical_symbols(), atoms.get_forces()):
            if forces_dict.get(sym, None) is None:
                forces_dict[sym] = [force]
            else:
                forces_dict[sym].append(force)
        
    return energies, forces_dict

def evaluate_all(frames, model_dirs):
    energies, forces_dict = read_properties(frames)

    from deepmd.calculator import DP
    for model_dir in model_dirs:
        dp_frames = frames.copy()
        pot = str(Path(model_dir) / 'graph.pb')
        calc = DP(model=pot)
        for atoms in frames:
            calc.reset()
            atoms.calc = calc
            dummy = atoms.get_forces()
        dp_ens, dp_fdict = read_properties(dp_frames)
        print(rms_dict(energies, dp_ens))
        for sym in forces_dict.keys():
            print(sym, rms_dict(forces_dict[sym], dp_fdict[sym]))
        exit()
        pass

    pass

if __name__ == '__main__':
    frames = read('/users/40247882/projects/oxides/dptrain/data.xyz', ':')
    ensemble_path = Path('/users/40247882/projects/oxides/dptrain/ensemble-2')
    model_dirs = []
    for p in ensemble_path.glob('model*'):
        model_dirs.append(p)
    model_dirs.sort()
    evaluate_all(frames, model_dirs)
    pass
