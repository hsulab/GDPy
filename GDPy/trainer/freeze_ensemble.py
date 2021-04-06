#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import subprocess
from pathlib import Path

import numpy as np

from ase.io import read, write


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

def check_finished(model_path):
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

def freeze_ensemble(ensemble_path):
    # find models
    model_dirs = []
    for p in ensemble_path.glob('model*'):
        model_dirs.append(p)
    model_dirs.sort()

    # freeze models
    for model_dir in model_dirs:
        #if check_finished(model_dir):
        #    freeze_model(model_dir)
        freeze_model(model_dir)

    return


if __name__ == '__main__':
    #ensemble_path = Path('/users/40247882/projects/oxides/gdp-main/it-0002/ensemble')
    ensemble_path = Path('/users/40247882/projects/oxides/gdp-main/it-0002/ensemble-more')
    freeze_ensemble(ensemble_path)
    pass
