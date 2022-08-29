#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

from ase.io import read, write
from ase.calculators.singlepoint import SinglePointCalculator

from GDPy.computation.worker.worker import DriverBasedWorker
from GDPy.potential.manager import PotManager
from GDPy.utils.command import parse_input_file

def register_worker(config_file: dict):
    """ return either potter or worker
    """
    params = parse_input_file(config_file)

    potter, driver, worker = None, None, None

    # - get potter first
    potential_params = params.get("potential", {})
    if not potential_params:
        potential_params = params
        # potential-only
    manager = PotManager()
    name = potential_params.get("name", None)
    potter = manager.create_potential(pot_name=name)
    potter.register_calculator(potential_params.get("params", {}))
    potter.version = potential_params.get("version", "unknown")

    # - try to get driver
    driver_params = params.get("driver", {})
    driver = potter.create_driver(driver_params) # use external backend

    # - scheduler
    scheduler_params = params.get("scheduler", {})
    if scheduler_params:
        potter.register_scheduler(scheduler_params)
    
    # - try worker
    if driver and potter.scheduler:
        worker = DriverBasedWorker(driver, potter.scheduler)
    
    return (potter if not worker else worker)

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
    driver, traj_dirs,
    traj_period, traj_fpath, traj_ind_fpath,
    include_first=False, include_last=True
):
    """ read trajectories from several directories
    """
    # - act, retrieve trajectory frames
    # TODO: more general interface not limited to dynamics
    # TODO: change this to joblib?
    # TODO: check whether the existed files are empty
    if not traj_fpath.exists():
        traj_indices = [] # use traj indices to mark selected traj frames
        all_traj_frames = []
        for t_dir in traj_dirs:
            # --- read confid and parse corresponding trajectory
            driver.directory = t_dir
            traj_frames = driver.read_trajectory()
            #print("n_trajframes: ", len(traj_frames))
            n_trajframes = len(traj_frames)
            # --- generate indices
            first, last = 0, n_trajframes-1
            # NOTE: last one should be always included since it may be converged structure
            cur_indices = list(range(0,len(traj_frames),traj_period))
            if include_last:
                if last not in cur_indices:
                    cur_indices.append(last)
            if not include_first:
                cur_indices = cur_indices[1:]
            # ----- map indices to global ones
            cur_nframes = len(all_traj_frames)
            cur_indices = [c+cur_nframes for c in cur_indices]
            # --- add frames
            traj_indices.extend(cur_indices)
            all_traj_frames.extend(traj_frames)
        np.save(traj_ind_fpath, traj_indices)
        write(traj_fpath, all_traj_frames)
    else:
        all_traj_frames = read(traj_fpath, ":")
    print("ntrajframes: ", len(all_traj_frames))
    #print(len(traj_indices))
            
    #print(traj_ind_fpath)
    if traj_ind_fpath.exists():
        traj_indices = np.load(traj_ind_fpath)
    all_traj_frames = [all_traj_frames[i] for i in traj_indices]
        #print(traj_indices)
    print("ntrajframes: ", len(all_traj_frames), f" by {traj_period} traj_period")

    return all_traj_frames

if __name__ == "__main__":
    pass