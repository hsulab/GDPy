#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os 
import argparse

import numpy as np 

from ase.io import read, write 
from ase.constraints import UnitCellFilter 
from ase.optimize import BFGS 

from deepmd.calculator import DP 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-c', '--cell', 
        help='bulk cell file'
    )
    parser.add_argument(
        '-m', '--model', 
        help='DP model'
    )
    parser.add_argument(
        '-w', '--work', 
        help='work directory'
    )

    args = parser.parse_args()

    #atoms = read('./CrO/CrO_mp-755073_primitive.cif')
    atoms = read(args.cell)

    calc = DP(
        #model="../merged_dataset/PES3/graph.pb", 
        model = args.model, 
        type_dict = {"O": 1, "Zn": 2, "Cr": 0}
    )

    cell = atoms.get_cell() 

    vols = []
    ens = []

    log_fname = os.path.join(args.work, 'DP/BM.log')
    with open(log_fname, 'w') as fopen:
        fopen.write('# START ')

    for x in np.linspace(0.95, 1.03, 9):
        atoms.set_calculator(calc)
        atoms.set_cell(cell * x, scale_atoms=True)
        traj = args.work + '/DP/cell_%s.traj' %str(x)

        # lattice optimisation at constant volume 
        ucf = UnitCellFilter(atoms, constant_volume=True)
        bfgs = BFGS(atoms, logfile=log_fname, trajectory=traj)

        bfgs.run(fmax=0.05, steps=500)
        
        # data
        en = atoms.get_potential_energy()

        ens.append(en/len(atoms))
        vols.append(atoms.get_volume()/len(atoms))

    content = '#  Volume (A^3/atom)       Energy (eV/atom)\n'

    for vol, en in zip(vols, ens):
        content += '  %12.8f  %12.8f  \n' %(vol, en)

    with open(os.path.join(args.work,'DP/energy_volume_dp.dat'), 'w') as fopen:
        fopen.write(content)
