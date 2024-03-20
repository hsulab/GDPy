#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
import itertools

from typing import List

import numpy as np

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
try:
    plt.style.use("presentation")
except Exception as e:
    #print("Used default matplotlib style.")
    ...

from ase import Atoms
from ase.io import read, write
from ase.calculators.singlepoint import SinglePointCalculator
from ase.utils.forcecurve import fit_raw

from .validator import AbstractValidator

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

def plot_results(images_dict: dict, prefix, wdir):
    """"""
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 8))
    plt.suptitle("Nudged Elastic Band Calculation")

    ax.set_xlabel("Reaction Coordinate [Å]")
    ax.set_ylabel("Potential Energy [eV]")

    #print(mcolors.TABLEAU_COLORS)
    for (name, images), c in zip(images_dict.items(), itertools.cycle(mcolors.TABLEAU_COLORS.keys())):
        ff = get_forcefit(images)

        ax.scatter(ff.path, ff.energies, color=c, facecolor="w", zorder=100)
        ax.plot(ff.fit_path, ff.fit_energies, "-", color=c, label=name)

    np.savetxt(wdir/f"{prefix}neb.dat", np.vstack((ff.fit_path, ff.fit_energies)).T, fmt="%8.4f")

    ax.legend()

    plt.savefig(wdir/f"{prefix}neb.png")

    return

class PathwayValidator(AbstractValidator):


    def run(self, dataset, worker = None, *args, **kwargs):
        """"""
        super().run()

        # - prediction
        #nebtraj = dataset.get("prediction", None)[0]
        nebtraj = dataset.get("prediction", None)
        self._irun(images=nebtraj, prefix="pre-", worker=worker)

        return
    
    def _irun(self, images: List[Atoms], prefix: str, worker):
        """"""
        self._print(f"{worker = }")
        # TODO: Different NEB pathway
        #       default number of images
        #       use its own number
        #       same number as reference
        worker.directory = self.directory/"_xxx"
        worker.run([images[0], images[-1]])
        worker.inspect(resubmit=True)
        if worker.get_number_of_running_jobs() == 0:
            ret = worker.retrieve(
                include_retrieved=True,
            ) # (npairs, nbands, nimages)
            #self._print(f"{ret = }")

        images_dict = dict(
            reference = images,
            prediction = ret[0][-1]
        )

        plot_results(images_dict, prefix=prefix, wdir=self.directory)

        return

    def _compare_results(self, images_dict: dict):
        """"""        

        raise NotImplementedError()


if __name__ == "__main__":
    ...
