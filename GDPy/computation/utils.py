#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

from ase.io import read, write
from ase.calculators.singlepoint import SinglePointCalculator

def create_single_point_calculator(atoms_sorted, resort, calc_name):
    """ create a spc to store calc results
        since some atoms may share a calculator
    """
    atoms = atoms_sorted.copy()[resort]
    calc = SinglePointCalculator(
        atoms,
        energy=atoms_sorted.get_potential_energy(),
        forces=atoms_sorted.get_forces(apply_constraint=False)[resort]
        # TODO: magmoms?
    )
    calc.name = calc_name
    atoms.calc = calc

    return atoms

def parse_type_list(atoms):
    """parse type list for read and write structure of lammps"""
    # elements
    type_list = list(set(atoms.get_chemical_symbols()))
    type_list.sort() # by alphabet

    return type_list


def read_trajectories(
    driver, traj_dirs, type_list, 
    traj_period, traj_fpath, traj_ind_fpath
):
    """ read trajectories from several directories
    """
    # - act, retrieve trajectory frames
    # TODO: more general interface not limited to dynamics
    # TODO: change this to joblib?
    if not traj_fpath.exists():
        traj_indices = [] # use traj indices to mark selected traj frames
        all_traj_frames = []
        for t_dir in traj_dirs:
            # --- read confid and parse corresponding trajectory
            driver.directory = t_dir
            traj_frames = driver.read_trajectory(type_list=type_list)
            # --- generate indices
            # NOTE: last one should be always included since it may be converged structure
            cur_nframes = len(all_traj_frames)
            cur_indices = list(range(0,len(traj_frames)-1,traj_period)) + [len(traj_frames)-1]
            cur_indices = [c+cur_nframes for c in cur_indices]
            # --- add frames
            traj_indices.extend(cur_indices)
            all_traj_frames.extend(traj_frames)
        np.save(traj_ind_fpath, traj_indices)
        write(traj_fpath, all_traj_frames)
    else:
        all_traj_frames = read(traj_fpath, ":")
    print("ntrajframes: ", len(all_traj_frames))
            
    if traj_ind_fpath.exists():
        traj_indices = np.load(traj_ind_fpath)
        all_traj_frames = [all_traj_frames[i] for i in traj_indices]
        #print(traj_indices)
    print("ntrajframes: ", len(all_traj_frames), f" by {traj_period} traj_period")

    return all_traj_frames