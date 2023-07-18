#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import matplotlib as mpl
mpl.use("Agg") #silent mode
from matplotlib import pyplot as plt
try:
    plt.style.use("presentation")
except Exception as e:
    ...

from ase.utils.forcecurve import fit_raw

def get_forcefit(images):
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

class ReactionComparator():

    def __init__(self) -> None:
        """"""

        return
    
    def run(self, prediction, reference):
        """"""
        # - input shape should be (2, ?, ?nimages_per_band)?
        print(f"reference: {reference}")
        print(f"prediction: {prediction}")

        reference = reference.get_marked_structures()
        prediction = prediction.get_marked_structures()

        dene = prediction[0].get_potential_energy() - reference[0].get_potential_energy()

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 8))
        plt.suptitle("Nudge Elastic Band Calculation")

        ax.set_xlabel("Reaction Coordinate [Å]")
        ax.set_ylabel("Potential Energy [eV]")

        self._add_axis(ax, reference, dene, "dft")
        self._add_axis(ax, prediction, 0., "dp")

        ax.legend()

        plt.savefig(self.directory/"neb.png")

        return
    
    def _add_axis(self, ax, images, dene: float, label):
        """"""
        ff = get_forcefit(images)
        ax.scatter(ff.path, ff.energies-dene, label=label)
        ax.plot(ff.fit_path, ff.fit_energies-dene, "k-")

        return


if __name__ == "__main__":
    ...