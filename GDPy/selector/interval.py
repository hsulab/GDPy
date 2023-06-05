#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Union, List

import numpy as np

from ase import Atoms

from GDPy.core.register import registers
from GDPy.data.trajectory import Trajectories
from GDPy.selector.selector import AbstractSelector


@registers.selector.register
class IntervalSelector(AbstractSelector):

    name = "interval"

    default_parameters = dict(
        traj_period = 1,
        include_first = True,
        include_last = True,
        number = [4, 0.2],
        random_seed = 1112
    )

    """This is a number-unaware selector.
    """

    def __init__(self, directory="./", *args, **kwargs) -> None:
        """"""
        super().__init__(directory, *args, **kwargs)

        return
    
    def _mark_structures(self, inp_dat, *args, **kargs) -> None:
        """Select structures.

        Add unmasks to input trajectories.

        Args:
            inp_dat: Structures.
        
        """
        trajectories = inp_dat

        for traj in trajectories:
            markers = traj.markers
            #print(markers)
            nstructures = len(markers)
            first, last = 0, nstructures-1
            if self.include_first:
                curr_indices = list(range(0,nstructures,self.traj_period))
                if self.include_last:
                    if last not in curr_indices:
                        curr_indices.append(last)
                else:
                    if last in curr_indices:
                        curr_indices.remove(last)
            else:
                curr_indices = list(range(1,nstructures,self.traj_period))
                if self.include_last:
                    if last not in curr_indices:
                        curr_indices.append(last)
                else:
                    if last in curr_indices:
                        curr_indices.remove(last)
            new_markers = [markers[i] for i in curr_indices]
            #print(new_markers)
            traj.markers = new_markers

        return

if __name__ == "__main__":
    ...