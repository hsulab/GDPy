#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob

from collections import namedtuple

import numpy as np

from ase.io import read, write 
from ase.io.trajectory import Trajectory 

from ase.constraints import UnitCellFilter
from ase.optimize import BFGS 

from dprss.utils import read_arrays

def check_convergence(atoms, fmax=0.05):
    """Check the convergence of the trajectory"""

    forces = atoms.get_forces()

    max_force = np.max(np.fabs(forces))

    converged = False
    if max_force < fmax:
        converged = True 

    return converged

def read_massive_trajs(work, prefix, indices, restart=False):
    """read many trajectories and extract info"""
    # traj pattern 
    traj_pattern = os.path.join(
        os.path.abspath(work), prefix
    )

    start, end = [int(i) if i != '' else None for i in indices.split(':')]
    if start is None:
        start = 0
    if end is not None:
        traj_fpaths = [traj_pattern+str(i).zfill(4)+'.traj' for i in range(start,end)]
    else:
        traj_fpaths = glob.iglob(traj_pattern+'*.traj')

    # data file 
    content = '# {:>20s}  {:>20s}  {:>20s}  {:>20s}  {:<}\n'.format(
        'index', 'energy', 'maxforce', 'steps', 'path'
    )
    if restart:
        count = 0
        with open('dynamics.dat', 'w') as fopen:
            fopen.write(content)
    else:
        count = start
        with open('dynamics.dat', 'a') as fopen:
            fopen.write(content)

    # data info
    for traj_fname in traj_fpaths:
        # read and store 
        num_idx = (os.path.basename(traj_fname).strip(prefix).split('.'))[0]
        traj = read(traj_fname, ':')

        # only choose dp converged trajectory
        final_atoms = traj[-1]
        #last_energy = final_atoms.get_potential_energy()[0] # unfixed interface
        last_energy = final_atoms.get_potential_energy() # unfixed interface
        last_maxforce = np.max(np.fabs(final_atoms.get_forces()))
        #print(count, last_energy, last_maxforce, len(traj), traj_fname)
        #print(count, last_energy.shape, last_maxforce.shape, len(traj), traj_fname)

        content = '  {:>20d}  {:>20.8f}  {:>20.4f}  {:>20d}  {:<}\n'.format(
            count, last_energy, last_maxforce, len(traj), traj_fname
        )
        with open('dynamics.dat', 'a') as fopen:
            fopen.write(content)
        count += 1

    return 

def select_minima(dyndat='./dynamics.dat', blztraj='./blz_trajs.dat', num_minima=500):
    """"""
    data = read_arrays(dyndat)
    converged_data = [line for line in data if float(line[2]) < 0.05]

    traj_fpaths = [i[-1] for i in converged_data]

    energies = [float(i[1]) for i in converged_data]

    blz_trajs, props = boltzmann_histogram_selection(
        energies, traj_fpaths, num_minima, kT=3.0
    )

    blz_trajs.sort(key=lambda fname:str( fname ))
    with open(blztraj, 'w') as fopen:
        fopen.write('#\n'+'\n'.join(blz_trajs))

    # merge frames 
    minima_only = True
    if minima_only:
        for traj_name in blz_trajs:
            frames = [read(traj_name, '-1')]
            for step, atoms in enumerate(frames):
                atoms.info['file'] = traj_name
                #atoms.info['step'] = step
            write('blz_frames.xyz', frames, append=True)

    return 


def configuration_property(atoms, prop_type='energy'):
    """ Calculate property for Boltzmann-biased Histogram Selection
        Currently, only potential energy is supported property. 
    """
    # TODO: calculate enthalpy at given external pressure 
    
    if prop_type == 'energy':
        prop = atoms.get_potential_energy()

    return prop 


def boltzmann_histogram_selection(props, traj_fpaths, num_minima, kT=-1.0):
    """"""
    # calculate minima properties 

    # compute desired probabilities for flattened histogram
    histo = np.histogram(props)
    min_prop = np.min(props)

    config_prob = []
    for H in props:
        bin_i = np.searchsorted(histo[1][1:], H)
        if histo[0][bin_i] > 0.0:
            p = 1.0/histo[0][bin_i]
        else:
            p = 0.0
        if kT > 0.0:
            p *= np.exp(-(H-min_prop)/kT)
        config_prob.append(p)

    selected_indices = []
    for i in range(num_minima):
        # TODO: rewrite by mask 
        config_prob = np.array(config_prob)
        config_prob /= np.sum(config_prob)
        cumul_prob = np.cumsum(config_prob)
        rv = np.random.uniform()
        config_i = np.searchsorted(cumul_prob, rv)
        #print(converged_trajectories[config_i][0])
        selected_indices.append(traj_fpaths[config_i])

        # remove from config_prob by converting to list
        config_prob = list(config_prob)
        del config_prob[config_i]

        # remove from other lists
        del props[config_i]
        del traj_fpaths[config_i]
        
    return selected_indices, props 


def select_trajectory(cwd, num_minima):
    """"""
    namedTraj = namedtuple(
        'namedTraj', ['traj', 'idx']
    )
    # traj pattern 
    traj_pattern = os.path.abspath(cwd) + '/ucf-*.traj'

    # read relaxation trajectoris
    traj_count = 0
    converged_trajectories = []
    for traj_fname in glob.iglob(traj_pattern):
        num_idx = (os.path.basename(traj_fname).split('.')[0]).split('-')[1]
        traj = read(traj_fname, ':')
        # only choose dp converged trajectory
        final_atoms = traj[-1]
        if check_convergence(final_atoms, 0.05):
            # add info
            for fidx, atoms in enumerate(traj):
                atoms.info['step'] = '%d-%d' %(traj_count, fidx)
            #print(num_idx, traj_fname)
            converged_trajectories.append( namedTraj(traj=traj, idx=traj_count) )
        else:
            pass
        traj_count += 1

    # get full trajectories leading to minima
    selected_ntrajs = boltzmann_histogram_selection(
        converged_trajectories, num_minima, kT=3.0
    )

    # get selected minima and related configurations from their trajectories 
    namedFrame = namedtuple(
        'namedFrame', ['atoms', 'traj', 'local_index', 'global_index']
    )
    total_frames = []
    selected_minima = [] # minima are selected unconditionally 

    count = 0
    for ntraj in selected_ntrajs: 
        total_frames.extend(ntraj.traj)
        count += 1
        pass

    write('selected_trajframes.xyz', total_frames)

    return 


if __name__ == '__main__':
    pass 
