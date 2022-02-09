#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from collections import namedtuple

from typing import NoReturn, Tuple, Union, Mapping, List

import numpy
import numpy as np

from ase.io import read, write 

import matplotlib as mpl
mpl.use('Agg') #silent mode
from matplotlib import pyplot as plt


def rms_dict(x_ref: List[float], x_pred: List[float]):
    """ Takes two datasets of the same shape 
        and returns a dictionary containing RMS error data
    """

    x_ref = np.array(x_ref)
    x_pred = np.array(x_pred)

    if np.shape(x_pred) != np.shape(x_ref):
        raise ValueError('WARNING: not matching shapes in rms')

    error_2 = (x_ref - x_pred) ** 2

    average = np.sqrt(np.average(error_2))
    std_ = np.sqrt(np.var(error_2))

    return {'rmse': average, 'std': std_}

def parity_plot(
        emulation: Tuple[str, Union[numpy.array,numpy.ndarray]], 
        reference: Tuple[str, Union[numpy.array,numpy.ndarray]], 
        ax, prop: Tuple[str, str] = ('energy','[eV/atom]'), title: str ='Parity Plot'
) -> NoReturn :
    """ Plots the distribution of energy per atom on the output vs the input"""
    # extract data
    xdata, ydata = emulation[1], reference[1]
    # scatter plot of the data
    ax.scatter(xdata, ydata)

    # get the appropriate limits for the plot
    pmax = np.max(np.array([xdata,ydata])) # property max
    pmin = np.min(np.array([xdata,ydata]))
    plim = (pmin - 0.05, pmax + 0.05)

    ax.set_xlim(plim)
    ax.set_ylim(plim)

    # add line of slope 1 for refrence
    ax.plot(plim, plim, c='k')

    # set labels
    ax.set_xlabel(prop[0]+' '+emulation[0]+' '+prop[1], fontsize=24)
    ax.set_ylabel(prop[0]+' '+reference[0]+' '+prop[1], fontsize=24)

    #set title
    ax.set_title(title, fontsize=24, fontweight='bold')

    # add text about RMSE
    _rms = rms_dict(xdata, ydata)
    rmse_text = 'RMSE:\n{:>.3f}+-{:>.3f} {:s}'.format(_rms['rmse'], _rms['std'], prop[1])
    ax.text(
        0.9, 0.1, rmse_text, transform=ax.transAxes, fontsize=24, fontweight='bold', 
        horizontalalignment='right', verticalalignment='bottom'
    )
    
    return

PropInfo = namedtuple(
    'PropInfo', ['xlabel', 'ylabel', 'title']
) 


def parity_plot_dict(
        emulation: Mapping[str, Union[numpy.array,numpy.ndarray]], 
        reference: Mapping[str, Union[numpy.array,numpy.ndarray]], 
        ax, prop_dict: dict = None, apex=[1.0], write_info=True
):
    """ Plots the distribution of energy per atom on the output vs the input"""
    if prop_dict is None:
        prop_dict = {}
        prop_dict['xlabel'] = 'x'
        prop_dict['ylabel'] = 'y'
        prop_dict['title'] = 'Parity Plot'
    
    rmse_results = {}

    rmse_text = ''
    names = emulation.keys()
    pmaxs, pmins = [], []
    for name in names:
        # extract data
        xdata, ydata = emulation[name], reference[name]
        # print(xdata)
        # print(ydata)
        # scatter plot of the data
        ax.scatter(xdata, ydata, label=name)
        # find max and min
        #print(np.max(xdata), np.max(ydata))
        pmaxs.append( np.max([np.max(xdata),np.max(ydata)]) )
        pmins.append( np.min([np.min(xdata),np.min(ydata)]) )
        # add text about RMSE
        _rms = rms_dict(xdata, ydata)
        rmse_results[name] = _rms
        rmse_text += 'RMSE:\n{:>.3f}+-{:>.3f} {:s}\n'.format(_rms['rmse'], _rms['std'], name)
    if write_info:
        ax.text(
            0.9, 0.1, rmse_text, transform=ax.transAxes, fontsize=24, fontweight='bold', 
            horizontalalignment='right', verticalalignment='bottom'
        )

    # get the appropriate limits for the plot
    pmax = np.max(pmaxs)
    pmin = np.min(pmins)
    plim = np.array((pmin - (pmax-pmin)*0.05, pmax + (pmax-pmin)*0.05))

    ax.set_xlim(plim)
    ax.set_ylim(plim)

    # add line of slope 1 for refrence
    for a in apex:
        if a == 1:
            ax.plot(plim, a*plim, c="k")
        else:
            ax.plot(plim, a*plim, ls="--", c="k", label = "${}\sigma$".format(a))

    # set labels
    ax.set_xlabel(prop_dict["xlabel"])
    ax.set_ylabel(prop_dict["ylabel"])

    #set title
    ax.set_title(prop_dict["title"])

    ax.legend()
    
    return rmse_results

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
    
    return

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
    pass