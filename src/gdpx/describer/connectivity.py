#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
from ase.data import atomic_numbers

from gdpx.geometry.spatial import (check_atomic_distances,
                                   get_bond_distance_dict)

from .describer import AbstractDescriber


class ConnectivityDescriber(AbstractDescriber):

    name: str = "connectivity"

    def __init__(self, covalent_ratio=[0.8, 2.0], *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.covalent_ratio = covalent_ratio

        return

    def run(self, structures):
        """"""
        # infer chemical species
        chemical_symbols = []
        for atoms in structures:
            chemical_symbols.extend(atoms.get_chemical_symbols())
        chemical_symbols = set(chemical_symbols)
        chemical_numbers = [atomic_numbers[s] for s in chemical_symbols]

        bond_distance_dict = get_bond_distance_dict(chemical_numbers)

        connectivity_states = []
        for atoms in structures:
            is_connected = check_atomic_distances(
                atoms,
                covalent_ratio=self.covalent_ratio,
                bond_distance_dict=bond_distance_dict,
                allow_isolated=False
            )
            connectivity_states.append(is_connected)

        return np.array(connectivity_states, dtype=np.int32)


if __name__ == "__main__":
    ...
