#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import itertools

import numpy as np

from ase.geometry import find_mic

from .comparator import AbstractComparator

from ..builder.group import create_a_group


class CartesianComparator(AbstractComparator):

    dtol_avg: float = 0.1 # displacement tolerance tolerance, Ang
    dtol_std: float = 0.02 # displacement tolerance tolerance, Ang

    mic: bool = True

    group: str = None

    def __init__(self, dtol_avg=0.1, dtol_std=0.02, mic=True, group=None, *args, **kwargs):
        """"""
        super().__init__(*args, **kwargs)

        self.dtol_avg = dtol_avg
        self.dtol_std = dtol_std

        self.mic = mic
        self.group = group

        return
    
    def looks_like(self, a1, a2) -> bool:
        """"""
        is_similar = False
        na1, na2 = len(a1), len(a2)
        if na1 == na2:
            c1, c2 = a1.get_cell(complete=True), a2.get_cell(complete=True)
            if np.allclose(c1, c2):
                ainds = None
                if self.group is not None:
                    g1 = create_a_group(a1, self.group)
                    g2 = create_a_group(a2, self.group)
                    if g1 == g2:
                        ainds = g1
                else:
                    ainds = range(na1) # atomic indices
                if ainds is not None:
                    self._print(f"{len(ainds)}")
                    pos1, pos2 = a1.positions[ainds, :], a2.positions[ainds, :]
                    # TODO: consider permutations?
                    #perms = itertools.permutations(range(len(ainds)))
                    #for p in perms:
                    #    self._print(p)
                    #    if self.mic:
                    #        vectors, distances = find_mic(pos1 - pos2[p, :], c1, pbc=True)
                    #    else:
                    #        vectors = pos1 - pos2[p, :]
                    #    disps = np.linalg.norm(vectors, axis=1)
                    #    self._print(disps)
                    #    davg = np.average(disps) 
                    #    dstd = np.sqrt(np.var(disps))
                    #    self._print(f"davg: {davg} dstd: {dstd}")
                    if self.mic:
                        vectors, distances = find_mic(pos1 - pos2, c1, pbc=True)
                    else:
                        vectors = pos1 - pos2
                    disps = np.linalg.norm(vectors, axis=1)
                    davg = np.average(disps) 
                    dstd = np.sqrt(np.var(disps))
                    if davg <= self.dtol_avg:
                        is_similar = True
            else:
                ...
        else:
            ...

        return  is_similar


if __name__ == "__main__":
    ...