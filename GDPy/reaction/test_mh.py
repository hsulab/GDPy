#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from ase import Atoms, Atom
from ase.build import fcc110
from ase.optimize.minimahopping import MinimaHopping
from ase.calculators.emt import EMT
from ase.constraints import FixAtoms, Hookean

# Make the Pt 110 slab.
atoms = fcc110('Pt', (2, 2, 2), vacuum=7.)

# Add the Cu2 adsorbate.
adsorbate = Atoms(
    [Atom('Cu', atoms[7].position + (0., 0., 2.5)),
     Atom('Cu', atoms[7].position + (0., 0., 5.0))]
)
atoms.extend(adsorbate)

# Constrain the surface to be fixed and a Hookean constraint between
# the adsorbate atoms.
constraints = [
    FixAtoms(indices=[atom.index for atom in atoms if
                                 atom.symbol == 'Pt']),
    Hookean(a1=8, a2=9, rt=2.6, k=15.),
    Hookean(a1=8, a2=(0., 0., 1., -15.), k=15.), 
]
atoms.set_constraint(constraints)

# Set the calculator.
calc = EMT()
atoms.calc = calc

# Instantiate and run the minima hopping algorithm.
hop = MinimaHopping(
    atoms,
    Ediff0=2.5,
    T0=4000.
)
hop(totalsteps=10)

from ase.optimize.minimahopping import MHPlot

mhplot = MHPlot()
mhplot.save_figure('summary.png')

if __name__ == '__main__':
    pass
