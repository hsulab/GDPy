#!/usr/bin/env python3
# -*- coding: utf-8 -*

import ase
from ase import Atoms
from ase.collections import g2
from ase.build import molecule

def build_species(species):
    # - build adsorbate
    atoms = None
    if species in ase.data.chemical_symbols:
        atoms = Atoms(species, positions=[[0.,0.,0.]])
    elif species in g2.names:
        atoms = molecule(species)
        #print("molecule: ", atoms.positions)
        #exit()
    else:
        raise ValueError(f"Cant create species {species}")
    
    return atoms