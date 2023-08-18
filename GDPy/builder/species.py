#!/usr/bin/env python3
# -*- coding: utf-8 -*

import time
from typing import List

import numpy as np

import ase
from ase import Atoms
from ase.io import read, write
from ase.collections import g2
from ase.build import molecule

from .builder import StructureBuilder


def compute_number_ideal_gas_molecules(pressure: float, volume: float, temperature: float):
    """Compute number of gas molecules by pV = nRT.

    1 eV/Ang3 = 160.21766208 GPa
    1 eV/Ang3 = 1581225.3844559586 atm
    1 atm = 1.01325 bar = 0.000101325 GPa
    1 atm = 6.32420912180121e-07 eV/Ang3
    1 Pa = 6.241509125883257e-12 eV/Ang3

    """
    #R = 8.314
    #nspecies = (pressure*volume) / (R*temperature) * 6.0221367e23
    kB = 1.38 * 1e-23 # J/K
    nmolecules = (pressure*volume) / (kB*temperature)

    return nmolecules


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


class MoleculeBuilder(StructureBuilder):

    name: str = "molecule"

    #: Default box for molecule to stay in. 
    default_box = np.eye(3)*20.

    #: Whether centre the molecule.
    recentre: bool = True

    def __init__(self, name: str=None, filename: str=None, box=None, recentre=True, directory="./", *args, **kwargs):
        """"""
        super().__init__(directory=directory, *args, **kwargs)

        if (name is None and filename is None) or (name is not None and filename is not None):
            raise RuntimeError(f"Either name or filname should be set for {self.name}.")
        if name is not None:
            self.molecule_name, self.func = name, self._build
        if filename is not None:
            self.filename, self.func = filename, self._read
        
        if box is not None:
            self.default_box = box
        self.recentre = recentre

        return
    
    def run(self, *args, **kwargs) -> List[Atoms]:
        """"""
        super().run(*args, **kwargs)

        atoms = self.func()

        # - check box
        #   NOTE: for molecule without cell,
        #         atoms.get_cell() returns [0.,0.,0.] while atoms.get_cell(complete=True)
        #         returns [1.,1.,1.]
        cell = atoms.get_cell()
        if np.sum(np.fabs(cell)) < 1e-8:
            # The box is too small.
            atoms.set_cell(self.default_box)
        else:
            ...
        
        # - centre the molecule
        cell = atoms.get_cell(complete=True)
        if self.recentre:
            box_centre = np.sum(cell, axis=0)/2.
            cop = np.average(atoms.get_positions(), axis=0)
            atoms.translate(box_centre-cop)
            ...

        return [atoms]
    
    def _read(self) -> Atoms:
        """"""
        frames = read(self.filename, ":")
        nframes = len(frames)
        assert nframes == 1, "Only one structure is accepted."

        return frames[0]
    
    def _build(self) -> Atoms:
        """"""
        molecule_name = self.molecule_name
        # - build adsorbate
        atoms = None
        if molecule_name in ase.data.chemical_symbols:
            atoms = Atoms(molecule_name, positions=[[0.,0.,0.]])
        elif molecule_name in g2.names:
            atoms = molecule(molecule_name)
        else:
            raise RuntimeError(f"Cant create molecule {molecule_name}")

        return atoms


if __name__ == "__main__":
    ...