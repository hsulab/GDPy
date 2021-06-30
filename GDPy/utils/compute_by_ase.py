#!/usr/bin/env python3

import os
import time
import json
import logging
import argparse

import numpy as np

from ase.io import read, write
from ase.calculators.vasp import Vasp2

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

# VASP environments
# pseudo 
pp_path = "/home/mmm0586/apps/vasp/PseudoPotential"
#pp_path = "/mnt/scratch/chemistry-apps/dkb01416/vasp/PseudoPotential"
if 'VASP_PP_PATH' in os.environ.keys():
    os.environ.pop('VASP_PP_PATH')
os.environ['VASP_PP_PATH'] = pp_path

# vdw 
vdw_envname = 'ASE_VASP_VDW'
vdw_path = "/home/mmm0586/apps/vasp/PseudoPotential"
#vdw_path = "/mnt/scratch/chemistry-apps/dkb01416/vasp/pot"
if vdw_envname in os.environ.keys():
    _ = os.environ.pop(vdw_envname)
os.environ[vdw_envname] = vdw_path

#VASP_COMMAND = 'mpirun -n 32 /mnt/scratch/chemistry-apps/dkb01416/vasp/installed/intel-2016/5.4.1-TS/vasp_std 2>&1 > vasp.out'
VASP_COMMAND = "mpirun -n 40 /home/mmm0586/apps/vasp/installed/5.4.1-TS/vasp_std 2>&1 > vasp.out", 

basic_vasp_params = {
    # INCAR 
    "nwrite": 2, 
    "istart": 0, 
    "lcharg": False, 
    "lwave": False, 
    "lorbit": 10,
    "npar": 4,
    "xc": "pbe",
    "encut": 400,
    "prec": "Normal",
    "ediff": 1E-5,
    "nelm": 180, 
    "nelmin": 6, 
    "ispin": 1,
    "ismear": 1,
    "sigma": 0.2,
    "algo": "Fast", 
    "lreal": "Auto", 
    "isym": 0, 
    "ediffg": -0.05,
    "nsw": 200,
    "ibrion": 2,
    "isif": 2,
    "potim": 0.02, 
    # KPOINTS
    "gamma": True
}


def generate_calculator(calculator, kpts, command, directory='vasp-outputs'):
    """turn a dict into a ase-calculator"""
    calc_params = basic_vasp_params.copy()
    calc_params.update(kpts = kpts)
    calc_params.update(command = command)
    calc_params.update(directory = directory)

    calc = calculator(**calc_params)

    return calc

def run_calculation(
        stru_file, indices, prefix, incar_list
    ):
    """"""
    # initialise few files
    for idx in range(len(incar_list)):
        with open('calculated_'+str(idx)+'.xyz', 'w') as writer:
            writer.write('')

    # read structures
    frames = read(stru_file, indices)
    start, end = indices.split(':')
    if start == '':
        start = 0
    else:
        start = int(start)
    logger.info('%d structures in %s from %d\n', len(frames), stru_file, start)

    # run calc
    for jdx, incar in enumerate(incar_list):
        logger.info(
            "\n===== Calculation Stage %d =====\n", jdx
        )
        for idx, atoms in enumerate(frames):
            logger.info(
                "Structure Number %d\n", idx+start
                #"Structure Number %d\n Index in XYZ is %d\n", idx, atoms.info['step']
            )
            # 
            kpts = np.linalg.norm(atoms.cell, axis=1).tolist()
            kpts = [int(20./k)+1 for k in kpts] 

            calc = generate_calculator(
                Vasp2, 
                kpts, 
                VASP_COMMAND,
                prefix+'_'+str(jdx)+'_'+str(idx+start)
            )
            atoms.calc = calc
            atoms.calc.read_incar(incar)
            dummy = atoms.get_forces() # just to call a calculation 
            write('calculated_'+str(jdx)+'.xyz', atoms, append=True)

    return

if __name__ == '__main__':
    logger.info(
        '\nStart at %s\n', 
        time.asctime( time.localtime(time.time()) )
    )

    # args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-s', '--structure', 
        default='example.xyz', 
        help='input structures stored in xyz format file'
    )
    parser.add_argument(
        '-p', '--parameter', 
        default='INCAR', nargs='*',
        help='calculator-related parameters in json format file'
    )
    parser.add_argument(
        '-i', '--indices', 
        default=':', 
        help='unsupported frame selection'
    )

    args = parser.parse_args()

    # run calculation 
    run_calculation(args.structure, args.indices, 'vasp', args.parameter)

    logger.info(
        '\nFinish at %s\n', 
        time.asctime( time.localtime(time.time()) )
    )
