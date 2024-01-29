#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from typing import List

import numpy as np
from scipy.spatial import distance_matrix

from ase import Atoms
from ase.io import read, write
from ase.neighborlist import NeighborList, natural_cutoffs

from ..builder.group import create_a_group
from .comparator import AbstractComparator


def switch_function(distances, r_cut, r_shift=0., nn: int=6, mm: int=None):
    """"""
    nn = (6 if nn is None else nn)
    mm = (nn*2 if mm is None else mm)

    scaled_distances = (distances - r_shift) / r_cut

    return (1 - scaled_distances**nn) / (1 - scaled_distances**mm)


def compute_self_coorditaion_number(positions):
    """"""
    dmat = distance_matrix(positions, positions)
    #print(dmat)

    sf = switch_function(dmat, r_cut=3.8, nn=8, mm=14)
    np.fill_diagonal(sf, 0.)
    coordination = np.sum(sf, axis=1)

    return coordination

def compute_coorditaion_number(positions_a, positions_b, r_cut, nn=None, mm=None):
    """NOTE: This does not substract the self-coordination."""
    dmat = distance_matrix(positions_a, positions_b)
    #print(dmat)

    sf = switch_function(dmat, r_cut=r_cut, nn=8, mm=14)
    coordination = np.sum(sf, axis=1)

    return coordination


class CoordinationComparator(AbstractComparator):

    def __init__(self, pairs=None, acn_rmse: float=0.02, acn_max: float=0.20, *args, **kwargs):
        """"""
        super().__init__(*args, **kwargs)

        # - check pairs that should be a List of 2-element Lists
        pairs = (pairs if pairs is not None else [])
        npairs = len(pairs)
        assert npairs > 0, "Need at least 1 pair."
        pairs_ = []
        for i, pair in enumerate(pairs):
            ngroups = len(pair)
            assert ngroups >= 2, f"Need at least two groups to define the pair {i}."
            pairs_.append(pair+[None]*(5-ngroups))
        self.pairs = pairs_

        self.acn_rmse = acn_rmse
        self.acn_max = acn_max

        return
    
    def prepare_data(self, frames: List[Atoms]):
        """"""
        coordinations = []
        for atoms in frames[:]:
            natoms = len(atoms)
            coordination = []
            for group_a, group_b, r_cut, nn, mm in self.pairs:
                indices_a = create_a_group(atoms, group_a)
                positions_a = atoms.positions[indices_a, :]
                indices_b = create_a_group(atoms, group_b)
                positions_b = atoms.positions[indices_b, :]
                curr_coordination = compute_coorditaion_number(
                    positions_a, positions_b, r_cut, nn, mm
                )
                coordination.extend(sorted(curr_coordination.tolist()))
            coordinations.append(coordination)
        coordinations = np.array(coordinations)

        return coordinations
    
    def looks_like(self, fp1, fp2):
        """Compare the fingerprints, here, the coordination number."""
        fp1, fp2 = np.array(fp1), np.array(fp2)

        assert fp1.shape == fp2.shape, f"{fp1.shape} vs. {fp2.shape} ..."
        nfeatures = fp1.shape[0]

        is_similar = False
        
        error = fp1 - fp2
        rmse = np.sqrt(np.sum(error**2)/nfeatures)
        if rmse <= self.acn_rmse and np.max(np.fabs(error)) <= self.acn_max:
            is_similar = True

        return is_similar


if __name__ == "__main__":
    ...
