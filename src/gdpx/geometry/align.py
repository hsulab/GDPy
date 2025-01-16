#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import copy

from ase import Atoms
from ase.geometry import find_mic


def wrap_traj(frames: list[Atoms]):
    """Align positions according to the first frame.

    This is necessary for computing physical quantities base on atomic positions 
    with periodic boundary conditions.

    NOTE:
        This only works for fixed cell systems.
    
    TODO:
        Variable cell systems?

    """
    cell = frames[0].get_cell(complete=True)
    nframes = len(frames)
    for i in range(1,nframes):
        prev_positions = frames[i-1].get_positions()
        curr_positions = frames[i].get_positions()
        shift = curr_positions - prev_positions
        curr_vectors, _ = find_mic(shift, cell, pbc=True)
        frames[i].positions = prev_positions + curr_vectors

    return frames


if __name__ == "__main__":
    ...
  
