#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import List

import numpy as np

from ase import Atoms
from ase.io import read, write

from GDPy.builder.builder import StructureGenerator

"""Sample small molecule.
"""

class HypercubeGenerator(StructureGenerator):

    def __init__(self, directory="./", *args, **kwargs):
        super().__init__(directory, *args, **kwargs)
    
    def run(self, *args, **kwargs) -> List[Atoms]:
        """"""

        return



if __name__ == "__main__":
    # We'd like to sample CO2 (a trimer) with different distances and angles
    atoms_ = Atoms(
        "CO2", positions=[[0.,0.,0.],[1.2,0.,0.],[-1.2,0.,0.]], cell=10.*np.eye(3),
        pbc=True
    )

    npoints = 8
    distances = np.linspace(0.8, 2.2, npoints)
    print(distances)

    nangles = 5
    angles = np.linspace(60, 180, nangles) / 180. * np.pi
    print(angles)

    frames = []
    for ang in angles:
        for i in range(npoints):
            for j in range(i,npoints):
                atoms = atoms_.copy()
                # -- O1
                atoms[1].position[0] = distances[i]
                # -- O2
                x, y = np.cos(ang)*distances[j], np.sin(ang)*distances[j]
                atoms[2].position = [x,y,0.]
                ...
                frames.append(atoms)
    print("nframes: ", len(frames))
    write("CO2-hypercube.xyz", frames)
    
    ...