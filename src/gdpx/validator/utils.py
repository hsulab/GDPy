#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
from ase import Atoms
from scipy.interpolate import make_interp_spline


def get_properties(frames: list[Atoms], other_props=[]):
    """Get properties of frames for comparison.

    Currently, only total energy and forces are considered.

    Returns:
        tot_symbols: shape (nframes,)
        tot_energies: shape (nframes,)
        tot_forces: shape (nframes,3)

    """
    tot_symbols, tot_energies, tot_forces = [], [], []

    for atoms in frames:  # free energy per atom
        # -- basic info
        symbols = atoms.get_chemical_symbols()
        tot_symbols.extend(symbols)

        # -- energy
        energy = atoms.get_potential_energy()
        tot_energies.append(energy)

        # -- force
        forces = atoms.get_forces(apply_constraint=False)
        tot_forces.extend(forces.tolist())


def smooth_curve(bins, points):
    """"""
    spl = make_interp_spline(bins, points, k=3)
    bins = np.linspace(bins.min(), bins.max(), 300)
    points = spl(bins)

    for i, d in enumerate(points):
        if d < 1e-6:
            points[i] = 0.0

    return bins, points


if __name__ == "__main__":
    ...

