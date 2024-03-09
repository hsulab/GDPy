#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy

import numpy as np

import matplotlib as mpl
mpl.use("Agg") #silent mode
from matplotlib import pyplot as plt
try:
    plt.style.use("presentation")
except Exception as e:
    #print("Used default matplotlib style.")
    ...

from ase import Atoms
from ase.io import read, write
from ase.calculators.singlepoint import SinglePointCalculator
from ase.utils.forcecurve import fit_raw

from gdpx.validator.validator import AbstractValidator

"""Validate reaction pathway using NEB.
"""

def make_clean_atoms(atoms_):
    """Create a clean atoms from the input."""
    atoms = Atoms(
        symbols=atoms_.get_chemical_symbols(),
        positions=atoms_.get_positions().copy(),
        cell=atoms_.get_cell().copy(),
        pbc=copy.deepcopy(atoms_.get_pbc())
    )

    return atoms

def get_forcefit(images):
    """"""
    energies = np.array([a.get_potential_energy() for a in images])
    emin = np.min(energies)
    forces = [a.get_forces(apply_constraint=True) for a in images]
    positions = [a.positions for a in images]
    cell, pbc = images[0].get_cell(complete=True), True
    #energies -= emin
    #rxn_coords = compute_rxn_coords(images)
    #ax.scatter(rxn_coords, energies, label="dp")
    ff = fit_raw(energies, forces, positions, cell, pbc) # ForceFit
    #print(ff)

    return ff

def plot_results(images, prefix, wdir):
    """"""
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 8))
    plt.suptitle("Nudged Elastic Band Calculation")

    ax.set_xlabel("Reaction Coordinate [Å]")
    ax.set_ylabel("Potential Energy [eV]")

    ff = get_forcefit(images)

    ax.scatter(ff.path, ff.energies)
    ax.plot(ff.fit_path, ff.fit_energies, "k--")

    np.savetxt(wdir/f"{prefix}neb.dat", np.vstack((ff.fit_path, ff.fit_energies)).T, fmt="%8.4f")

    #ax.legend()

    plt.savefig(wdir/f"{prefix}neb.png")

    return

class PathwayValidator(AbstractValidator):


    def run(self, dataset, worker = None, *args, **kwargs):
        """"""
        super().run()

        # - prediction
        nebtraj = dataset.get("prediction", None)[0]
        self._irun(images=nebtraj, prefix="pre-")

        return
    
    def _irun(self, images, prefix):
        """"""
        plot_results(images=images, prefix=prefix, wdir=self.directory)

        return

    def _compare_results(self, images_dict: dict):
        """"""        

        raise NotImplementedError()


if __name__ == "__main__":
    ...