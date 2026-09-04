from typing import Any, Optional

import numpy as np
from ase.data import atomic_numbers

from gdpx.geometry.restraints import evaluate_restraints, parse_restraints
from gdpx.geometry.spatial import check_atomic_distances, get_bond_distance_dict

from .describer import BaseDescriber


class ConnectivityDescriber(BaseDescriber):
    name: str = "connectivity"

    def __init__(
        self,
        covalent_ratio=[0.8, 2.0],
        restraints: Optional[list[dict[str, Any]]] = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.covalent_ratio = covalent_ratio
        self.restraints = parse_restraints(restraints, covalent_ratio=covalent_ratio)

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
            is_connected = evaluate_restraints(atoms, self.restraints)
            if is_connected:
                is_connected = check_atomic_distances(
                    atoms,
                    covalent_ratio=self.covalent_ratio,
                    bond_distance_dict=bond_distance_dict,
                    restraints=self.restraints,
                    allow_isolated=False,
                )
            connectivity_states.append(is_connected)

        return np.array(connectivity_states, dtype=np.int32)
