#!/usr/bin/env python3

import os
import time
import json

from ase.io import read, write
from ase.calculators.vasp import Vasp2

from dprss.utils import parse_indices

"""
# logger
logLevel = logging.INFO

logger = logging.getLogger(__name__)
logger.setLevel(logLevel)

fh = logging.FileHandler(filename='log.txt', mode='w')
fh.setLevel(logLevel)

ch = logging.StreamHandler()
ch.setLevel(logLevel)

logger.addHandler(ch)
logger.addHandler(fh)
"""

# VASP environments

def parse_params(json_file):
    """pass json format params"""
    # read json
    with open(json_file, 'r') as fr:
        params = json.load(fr)

    # frames 
    indices = params.get('frames', None)
    if indices:
        start, end = parse_indices(indices)
    else:
        start, end = 0, '' 
    idx_tuple = (str(start),str(end))

    # calculation 
    calculation = params.get('calculation', None)
    calc_name = calculation.get('calc', None)
    calc_params = calculation.pop('params', None)
    print(calc_name)
    if calc_name == 'EMT':
        # no other params for emt 
        from ase.calculators.emt import EMT 
        calc_type = EMT
    elif calc_name == 'VASP':
        if calc_params:
            if type(calc_params) == dict:
                calc_params = [calc_params]
            else:
                if type(calc_params) == list:
                    pass
                else:
                    raise ValueError('Cannot read calc_params in %s' %(json_file))
        else:
            pass
    elif calc_name == 'DFTB':
        pass
    else:
        raise ValueError('unknown calculator')

    return idx_tuple, calc_type, calc_params

def generate_calculator(calculator, command, params, directory='vasp-outputs'):
    """turn a dict into a ase-calculator"""
    calc_params = params
    calc_params.update(command = command)
    calc_params.update(directory = directory)

    calc = calculator(**calc_params)

    return calc

def run_calculation(
        start, frames, prefix, calc, calc_params_list
    ):
    """"""
    for jdx in range(len(calc_params_list)):
        with open('calculated_'+str(jdx)+'.xyz','w') as fopen:
            fopen.write('')
    # initialise few files
    for jdx, calc_params in enumerate(calc_params_list):
        for idx, atoms in enumerate(frames):
            calc.reset() 
            atoms.set_calculator(calc) # TODO: generate specific calculator 
            dummy = atoms.get_forces() # just to call a calculation 
            write('calculated_'+str(jdx)+'.xyz', atoms, append=True)

        frames = read('calculated_'+str(jdx)+'.xyz', ':')
        if type(frames) != list:
            frames = [frames]

    return

def validate_function(xyzfile, param_json):
    # read calc params
    idx_tuple, calc_type, calc_params = parse_params(param_json) 

    # read structures
    frames = read(xyzfile, ':'.join(idx_tuple))

    # run calculation 
    run_calculation(idx_tuple[0], frames, 'dummy', calc_type(), ['dummy'])

    pass

if __name__ == '__main__':
    pass 
