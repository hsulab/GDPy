#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
from typing import Optional, List

import numpy as np

from ase import Atoms
from ase import units
from ase.io import read, write
from ase.build import make_supercell

from .builder import StructureModifier


def compute_density(atoms: Atoms):
    """Compute density in [g/cm^3]."""
    mass = np.sum(atoms.get_masses())
    
    volumes = atoms.get_volume()
    density = (mass/units._Nav) / (volumes*1e-24)  # a.u./A^3 -> g/cm^3

    return density


class ScaleModifier(StructureModifier):

    """Make a supercell and scale its box.
    """

    def __init__(self, supercell: Optional[List[int]]=None, box: Optional[List[float]]=None, density: float=-1.0, substrates=None, *args, **kwargs):
        """"""
        super().__init__(substrates=substrates, *args, **kwargs)

        self.supercell = np.array(supercell if supercell else [1,1,1])

        box_ = np.array(box)
        if box_.size == 3:
            self.box = np.diag(box_)
        elif box_.size == 9:
            self.box = np.reshape(box_, (3,3))
        else:
            raise RuntimeError(f"Unknown box `{box}`.")
        
        self.density = density  # [g/cm^3]

        return
    
    def run(self, substrates=None, size: int=1, *args, **kwargs) -> List[Atoms]:
        """"""
        super().run(substrates=substrates, *args, **kwargs)

        frames = []
        for substrate in self.substrates:
            curr_frames = self._irun(substrate=substrate, size=size, *args, **kwargs)
            frames.extend(curr_frames)

        return frames
    
    def _irun(self, substrate: Atoms, size: int, *args, **kwargs) -> List[Atoms]:
        """"""
        new_atoms = make_supercell(substrate, np.diag(self.supercell))
        # if self.cubic:  # TODO: This works only for an ortho box.
        #     new_cell = np.diag([np.max(new_atoms.cell.cellpar()[:3])]*3)
        #     new_atoms.set_cell(new_cell, scale_atoms=True)
        if self.box is not None:
            new_atoms.set_cell(self.box, scale_atoms=True)
        if self.density > 0.:
            prev_density = compute_density(new_atoms)
            new_cell = new_atoms.get_cell(complete=True)/((self.density/prev_density)**(1/3))
            new_atoms.set_cell(new_cell, scale_atoms=True)
            curr_density = compute_density(new_atoms)
            assert np.isclose(self.density, curr_density), f"New density `{curr_density}` does not equal `{self.density}`."

        return [new_atoms]
    

if __name__ == "__main__":
    ...
