#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import pickle 

import numpy as np

from ase.io import read, write 

import matplotlib as mpl
mpl.use('Agg') #silent mode
from matplotlib import pyplot as plt


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

def energy_plot(ener_in, ener_out, ax, title='Plot of energy'):
    """ Plots the distribution of energy per atom on the output vs the input"""
    # scatter plot of the data
    ax.scatter(ener_in, ener_out)

    # get the appropriate limits for the plot
    for_limits = np.array(ener_in +ener_out)
    elim = (for_limits.min() - 0.05, for_limits.max() + 0.05)
    ax.set_xlim(elim)
    ax.set_ylim(elim)

    # add line of slope 1 for refrence
    ax.plot(elim, elim, c='k')

    # set labels
    ax.set_ylabel('energy by GAP / eV')
    ax.set_xlabel('energy by VASP / eV')

    #set title
    ax.set_title(title)

    # add text about RMSE
    _rms = rms_dict(ener_in, ener_out)
    rmse_text = 'RMSE:\n' + str(np.round(_rms['rmse'], 3)) + ' +- ' + str(np.round(_rms['std'], 3)) + 'eV/atom'
    ax.text(0.9, 0.1, rmse_text, transform=ax.transAxes, fontsize='large', \
            horizontalalignment='right', verticalalignment='bottom')

def force_plot(in_force, out_force, ax, symbol='H', title='Plot of force'):
    """ Plots the distribution of force components per atom 
        on the output vs the input only plots for the given atom type(s)"""
    # extract data for only one species
    in_force = in_force[symbol]
    out_force = out_force[symbol]

    # scatter plot of the data
    ax.scatter(in_force, out_force)

    # get the appropriate limits for the plot
    for_limits = np.array(in_force + out_force)
    flim = (for_limits.min() - 1, for_limits.max() + 1)
    ax.set_xlim(flim)
    ax.set_ylim(flim)

    # add line of
    ax.plot(flim, flim, c='k')

    # set labels
    ax.set_ylabel('force by GAP / (eV/Å)')
    ax.set_xlabel('force by VASP / (eV/Å)')

    #set title
    ax.set_title(title)

    # add text about RMSE
    _rms = rms_dict(in_force, out_force)
    rmse_text = 'RMSE:\n' + str(np.round(_rms['rmse'], 3)) + ' +- ' + str(np.round(_rms['std'], 3)) + 'eV/Å'
    ax.text(0.9, 0.1, rmse_text, transform=ax.transAxes, fontsize='large', horizontalalignment='right',
        verticalalignment='bottom')


def extract_energy_and_forces(atom_frames,calc=None,atomic=True):
    """
    Electronic free energy and Hellman-Feynman forces
    """
    energies_dft, forces_dft = [], {}
    energies_gap, forces_gap = [], {}

    for atoms in atom_frames: # free energy per atom
        # basic info
        symbols = atoms.get_chemical_symbols()
        if atomic:
            natoms = len(atoms)
        else:
            natoms = 1
        # energy
        free_energy = atoms.get_potential_energy(force_consistent=True) # electronic free energy
        energies_dft.append(free_energy/natoms)
        # force
        forces = atoms.get_forces()
        for sym, force in zip(symbols,forces):
            if sym in forces_dft.keys():
                forces_dft[sym].extend(force)
            else:
                forces_dft[sym] = list(force)
        if calc:
            # use quip to calculate gap predicted energy
            atoms.set_calculator(calc)
            free_energy = atoms.get_potential_energy(force_consistent=True) # electronic free energy
            energies_gap.append(free_energy/natoms)
            # force
            forces = atoms.get_forces()
            for sym, force in zip(symbols,forces):
                if sym in forces_gap.keys():
                    forces_gap[sym].extend(force)
                else:
                    forces_gap[sym] = list(force)

    return forces_dft, forces_gap, energies_dft, energies_gap


if __name__ == '__main__':
    # parser
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-t', '--train', 
        default='evaluated.xyz', help='trained structures in xyz format'
    )
    parser.add_argument(
        '-c', '--calc', action='store_true', 
        help='if using dp'
    )
    parser.add_argument(
        '-g', '--graph', 
        default='graph.pb', help='deep potential'
    )

    args = parser.parse_args()

    # calculate using dp 
    if args.calc:
        frames = read(args.train, ':')

        from deepmd.calculator import DP 
        calc = DP(
            model = args.graph, 
            #type_dict = {"O": 1, "Zn": 2, "Cr": 0}
        )
        data_dict = {'ener': [[],[]], 'force': [[],[]]}
        for atoms in frames:
            natoms = len(atoms) 
            energy = atoms.get_potential_energy() / natoms
            forces = atoms.get_forces() 
            data_dict['ener'][0].append(energy)
            data_dict['force'][0].extend(forces.flatten().tolist())
            # calculation 
            calc.reset() 
            atoms.calc = calc 
            energy = atoms.get_potential_energy() / natoms
            forces = atoms.get_forces()
            data_dict['ener'][1].append(energy)
            data_dict['force'][1].extend(forces.flatten().tolist())
            #print(dp_data)
        #print(tot_dat)

        with open('prop.pkl', 'wb') as fopen:
            pickle.dump(data_dict, fopen)

        print('finish calculating...')
    else:
        with open('prop.pkl', 'rb') as fopen:
            tot_data = pickle.load(fopen)
        ener_dat = tot_data['ener']
        force_dat = tot_data['force']
        force_dat[0] = {'Cu': force_dat[0]}
        force_dat[1] = {'Cu': force_dat[1]}
    
        # plot
        fig, axarr = plt.subplots(
            nrows=1, ncols=2, 
            gridspec_kw={'hspace': 0.3}, figsize=(12,8)
        )
        axarr = axarr.flat[:]

        plt.suptitle('Electronic Free Energy and Hellman-Feynman Forces')

        # plot energy
        energy_plot(
            ener_dat[0], ener_dat[1], 
            axarr[0], 'Energy on training data'
        )
        force_plot(
            force_dat[0], force_dat[1], 
            axarr[1], 'Cu', 'Force on training data - Cu'
        )

        plt.savefig('benchmark.png')

