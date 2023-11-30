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

from ..core.node import AbstractNode

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

class ReactionComparator(AbstractNode):

    def __init__(self, nimages=7, *args, **kwargs) -> None:
        """"""

        self.nimages = nimages

        return
    
    def run(self, prediction, reference):
        """"""
        # - input shape should be (2, ?, ?nimages_per_band)?
        print(f"reference: {reference}")
        print(f"prediction: {prediction}")

        reference = reference.get_marked_structures()
        prediction = prediction.get_marked_structures()

        nimages = self.nimages
        for i in range(int(len(reference)/nimages)):
            self._irun(
                prediction[i*nimages:(i+1)*nimages], 
                reference[i*nimages:(i+1)*nimages], 
                f"{i}".zfill(4)+"."
            )

        self._report()

        return

    def _irun(self, prediction, reference, prefix):
        dene = prediction[0].get_potential_energy() - reference[0].get_potential_energy()

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 8))
        plt.suptitle("Nudged Elastic Band Calculation")

        ax.set_xlabel("Reaction Coordinate [Å]")
        ax.set_ylabel("Potential Energy [eV]")

        self._add_axis(ax, reference, dene, "dft")
        self._add_axis(ax, prediction, 0., "dp")

        ax.legend()

        plt.savefig(self.directory/f"{prefix}neb.png")

        return
    
    def _add_axis(self, ax, images, dene: float, label):
        """"""
        ff = get_forcefit(images)
        ax.scatter(ff.path, ff.energies-dene, label=label)
        ax.plot(ff.fit_path, ff.fit_energies-dene, "k-")

        return
    
    def _report(self):
        """"""
        from reportlab.platypus import SimpleDocTemplate, Image, Paragraph

        story = []

        # - find figures
        nebfigures = list(self.directory.glob("*neb.png"))
        nebfigures = sorted(nebfigures)
        print(nebfigures)

        for i, p in enumerate(nebfigures):
            story.append(Paragraph(f"pathway {i}..."))
            image = Image(p)
            image.drawWidth = 240
            image.drawHeight = 160
            story.append(image)

        doc = SimpleDocTemplate(str(self.directory/"report.pdf"))
        doc.build(story)

        return


if __name__ == "__main__":
    ...