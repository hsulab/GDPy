#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os 
import time 
import json 
import shutil 

from ase.io import read, write 
from ase.io.trajectory import Trajectory
from ase.calculators.emt import EMT

import ase.constraints
from ase.constraints import FixAtoms

import ase.optimize
#from ase.constraints import UnitCellFilter
#from ase.optimize import BFGS 

from deepmd.calculator import DP 

def calc_props():
    # === stage 1. dft calculation 
    frames = read('selected_structures.xyz', ':')
    for atoms in frames:
        calc = EMT() # define here, avoid multiple usage
        atoms.set_calculator(calc)
        dummy = atoms.get_forces()
    write('calculated_structures.xyz', frames)

def dyn_param_parser(param_json):
    """"""
    with open(param_json) as fopen:
        params = json.load(fopen)

    dp_params = params['calculator']['params']
    dyn_params = params['dynamics']

    return dp_params, dyn_params

def run_dpdyn(selected_structures, cdw):
    # check log filename 
    time_string = time.strftime("%Y%m%d%H%M%S", time.localtime())
    log_fname = os.path.join(cdw, 'dyn.log')
    if os.path.exists(log_fname):
        shutil.copy(log_fname, os.path.join(cdw, 'bak.dyn.log_'+time_string))

    param_json = os.path.join(cdw, 'dyn.json')

    calc_params, dyn_params = dyn_param_parser(param_json)

    frame_index = dyn_params.pop("frames", None)
    if frame_index is not None:
        start, end = frame_index.split(':')
    else:
        start = 0
        frame_index = ':'

    frames = read(selected_structures, frame_index)
    content = "Number of Random Structures %8d\n\n" %(len(frames))

    with open(log_fname, 'w') as fopen:
        fopen.write(content)

    # share a calculator 
    # for some, it may fail in checking the system changes 
    calc = DP(**calc_params)

    # dynamics 
    cons = dyn_params.pop('constraints', None)
    filter_method = dyn_params.pop('filter', None)

    for idx, atoms in enumerate(frames):
        with open(log_fname, 'a') as fopen:
            fopen.write('Structure Index %8d\n' %(idx+int(start)))

        # set dp calculator
        calc.reset() # clear all info from last calculated frame 
        atoms.calc = calc # use new object method 

        tname = os.path.join(cdw, 'ucf-%s.traj' %(str(idx+int(start)).zfill(4)))

        props = ['energy', 'forces', 'stress']
        traj = Trajectory(tname, 'w', atoms, props)

        # check fixed atoms 
        if cons is not None:
            # TODO: better fixed atoms command 
            fix_command = cons['FixAtoms']
            if type(fix_command) == dict:
                height = fix_command.get('Z', None)
                if height is not None:
                    indices = [
                        atom.index for atom in atoms if atom.position[2] < height
                    ]
                else:
                    raise ValueError('error in FixAtoms')
            else:
                indices = fix_command 
            with open(log_fname, 'a') as fopen:
                fopen.write('FixedAtoms %8d\n' %(len(indices)))
            fa_cons = FixAtoms(indices = indices)
            atoms.set_constraint(fa_cons)

        # relaxation with cell 
        filtered_atoms = atoms
        if filter_method is not None: 
            filter_method = getattr(ase.constraints, dyn_params['filter'])
            filtered_atoms = filter_method(atoms)
        optimiser = getattr(ase.optimize, dyn_params['optimizer']['method'])
        opt = optimiser(filtered_atoms, logfile=log_fname, trajectory=traj)
        opt.run(**dyn_params['optimizer']['params'])

    return

if __name__ == '__main__':
    pass 
