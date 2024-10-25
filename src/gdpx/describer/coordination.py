#!/usr/bin/env python3
# -*- coding: utf-8 -*


import itertools
from typing import List, Optional

import numpy as np
import numpy.typing as npt
from ase import Atoms
from ase.io import read, write
from ase.neighborlist import neighbor_list
from scipy.spatial import distance_matrix

from .describer import AbstractDescriber


def switch_function(
    distances: npt.NDArray,
    r_cut: float,
    r_shift: float = 0.0,
    nn=6,
    mm: Optional[int] = None,
):
    """"""
    nn = 6 if nn is None else nn
    mm = nn * 2 if mm is None else mm

    scaled_distances = (distances - r_shift) / r_cut

    return (1 - scaled_distances**nn) / (1 - scaled_distances**mm)


def compute_coordination_number(atoms: Atoms, r_cut: float, type_list: List[str]):
    """"""
    chemical_symbols = atoms.get_chemical_symbols()
    num_types = len(type_list)

    # i is in the ascending order
    i, j, d = neighbor_list("ijd", atoms, cutoff=r_cut)

    data = []
    for k, v in itertools.groupby(zip(i, j, d), key=lambda x: x[0]):
        # TODO: use pair-specific r_cut?
        v = np.bincount(
            [type_list.index(chemical_symbols[x[1]]) for x in v], minlength=num_types
        )
        data.append([type_list.index(chemical_symbols[k]), *v])
    data = np.array(data, dtype=np.int32)

    return data


def compute_coordination_number_statistics(data, cnmax: int, type_list: List[str]):
    """

    Args:
        cnmax: Maximum coordination number.

    """
    bins = np.arange(cnmax + 1, dtype=np.int32)

    hist = []
    num_types = len(type_list)
    for i in range(num_types):
        for j in range(num_types):
            pair_data = data[data[:, 0] == i, j + 1]
            hists_, edges_ = np.histogram(
                pair_data, bins=bins, density=True
            )
            hist.append([np.mean(pair_data), *hists_])
    hist = np.array(hist)

    return hist


class CoordinationDescriber(AbstractDescriber):

    name: str = "coordination"

    def __init__(self, r_cut: float, type_list: List[str], *args, **kwargs):
        """"""
        super().__init__(*args, **kwargs)

        self.r_cut = r_cut
        self.type_list = type_list

        return

    def run(self, structures):
        """"""
        self._print(f"{structures =}")

        # compute data along trajectory
        hist_data = []
        for atoms in structures:
            data = compute_coordination_number(
                atoms, r_cut=self.r_cut, type_list=self.type_list
            )
            hist = compute_coordination_number_statistics(
                data, cnmax=10, type_list=self.type_list
            )
            hist_data.append(hist)
        hist = np.mean(hist_data, axis=0)

        # save data
        bins = np.arange(hist.shape[1]-1)
        print("   avg  " + ("  CN{:>2d}  " * bins.shape[0]).format(*bins))
        for hists_ in hist:
            print(("{:>6.4f}  " * hists_.shape[0]).format(*hists_))

        return


if __name__ == "__main__":
    ...
