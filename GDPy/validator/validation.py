#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pathlib

import numpy as np

import matplotlib
matplotlib.use('Agg') #silent mode
import matplotlib.pyplot as plt

from ase import Atoms
from ase.io import read, write

from GDPy.calculator.dp import DP

def xyz2dis(frames):
    """turn xyz into dimer data"""
    data = []
    for atoms in frames:
        # donot consider minimum image
        distance = np.linalg.norm(atoms[0].position-atoms[1].position) 
        energy = atoms.get_potential_energy()
        data.append([distance,energy])
    data = np.array(data)
    
    return np.array(data[:,0]), np.array(data[:,1])

def plot_dimer(task_name, distances, energies: dict, pname):
    """"""
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12,8))
    ax.set_title(
        task_name,
        fontsize=20, 
        fontweight='bold'
    )
    
    ax.set_xlabel('Distance [Å]', fontsize=16)
    ax.set_ylabel('Energyr [eV]', fontsize=16)

    for name, en in energies.items():
        ax.scatter(distances, en, label=name)
    ax.legend()

    plt.savefig(pname)

    return

def xyz2eos(frames):
    """turn xyz into eos data"""
    data = []
    for atoms in frames:
        # donot consider minimum image
        vol = atoms.get_volume()
        energy = atoms.get_potential_energy()
        data.append([vol,energy])
    data = np.array(data)
    
    return np.array(data[:,0]), np.array(data[:,1])

def run_calculation(frames, calc):
    dp_energies = []
    for atoms in frames:
        calc.reset()
        atoms.calc = calc
        dp_energies.append(atoms.get_potential_energy())
    dp_energies = np.array(dp_energies)

    return dp_energies

def validation(tasks, calc):
    """
    lattice constant
    equation of state
    """

    # run over various validations
    for validation, systems in tasks.items():
        for stru_path in systems:
            stru_path = pathlib.Path(stru_path)
            stru_name = stru_path.stem
            pic_path = stru_path.parent / (stru_name+'-dpx.png')
            print(pic_path)
            # run dp calculation
            frames = read(stru_path, ':')
            if validation == 'dimer':
                volumes, dft_energies = xyz2dis(frames)
            elif validation == 'volume':
                volumes, dft_energies = xyz2eos(frames)
            dp_energies = run_calculation(frames, calc)

            energies = {
                'reference': dft_energies, 
                'learned': dp_energies
            }

            # plot comparison
            plot_dimer(validation, volumes, energies, pic_path)

    return 


if __name__ == '__main__':
    tasks = {
        'dimer': [
            '/users/40247882/projects/oxides/dimers/O2-dimer.xyz',
            '/users/40247882/projects/oxides/dimers/Pt2-dimer.xyz'
        ],
        'volume': [
            '/users/40247882/projects/oxides/bulk/Pt-bulk.xyz',
            '/users/40247882/projects/oxides/bulk/PtO-bulk.xyz',
            '/users/40247882/projects/oxides/bulk/aPtO2-bulk.xyz',
            '/users/40247882/projects/oxides/bulk/bPtO2-bulk.xyz',
            '/users/40247882/projects/oxides/bulk/Pt3O4-bulk.xyz'
        ]
    }

    # set DP calculator
    type_map = {'O': 0, 'Pt': 1}
    model = '/users/40247882/projects/oxides/gdp-main/it-0005/ensemble/model-0/graph.pb'
    #model = '/users/40247882/projects/oxides/gdp-main/it-0002/ensemble-more/model-0/graph.pb'
    calc = DP(model=model, type_dict=type_map)

    validation(tasks, calc)
