#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import List

import numpy as np
from scipy.interpolate import make_interp_spline, BSpline

from ase import Atoms
from ase.geometry import find_mic

def get_properties(frames: List[Atoms], other_props = []):
    """Get properties of frames for comparison.

    Currently, only total energy and forces are considered.

    Returns:
        tot_symbols: shape (nframes,)
        tot_energies: shape (nframes,)
        tot_forces: shape (nframes,3)

    """
    tot_symbols, tot_energies, tot_forces = [], [], []

    for atoms in frames: # free energy per atom
        # -- basic info
        symbols = atoms.get_chemical_symbols()
        tot_symbols.extend(symbols)

        # -- energy
        energy = atoms.get_potential_energy() 
        tot_energies.append(energy)

        # -- force
        forces = atoms.get_forces(apply_constraint=False)
        tot_forces.extend(forces.tolist())

    return tot_symbols, tot_energies, tot_forces

def wrap_traj(frames):
    """Align positions according to the first frame.

    This is necessary for computing physical quantities base on atomic positions 
    with periodic boundary conditions.

    NOTE:
        This only works for fixed cell systems.
    
    TODO:
        Variable cell systems?

    """
    cell = frames[0].get_cell(complete=True)
    nframes = len(frames)
    for i in range(1,nframes):
        prev_positions = frames[i-1].get_positions()
        curr_positions = frames[i].get_positions()
        shift = curr_positions - prev_positions
        curr_vectors, curr_distances = find_mic(shift, cell, pbc=True)
        frames[i].positions = prev_positions + curr_vectors

    return frames

def smooth_curve(bins, points):
    """"""
    spl = make_interp_spline(bins, points, k=3)
    bins = np.linspace(bins.min(), bins.max(), 300)
    points= spl(bins)

    for i, d in enumerate(points):
        if d < 1e-6:
            points[i] = 0.0

    return bins, points


if __name__ == "__main__":
    ...