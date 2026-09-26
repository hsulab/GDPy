#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import numpy.typing
from ase import Atoms

from gdpx.utils.atoms_tags import get_tags_per_species

from .describer import BaseDescriber


def compute_formation_energy(
    atoms: Atoms, chempot_dict: dict[str, float]
) -> float:
    """"""
    identities = get_tags_per_species(atoms)
    identity_stats = {}
    for k, v in identities.items():
        identity_stats[k] = len(v)

    energy = atoms.get_potential_energy()

    formation_energy = energy - np.sum(
        [chempot_dict[k] * v for k, v in identity_stats.items()]
    )

    return formation_energy


class FormationEnergyDescriber(BaseDescriber):
    """This class describes the formation energy of a structure."""

    def __init__(self, chempot: dict[str, float], *args, **kwargs):
        """"""
        super().__init__(*args, **kwargs)

        self.chempot = chempot

        return

    def run(self, structures) -> numpy.typing.NDArray:
        """This method describes the formation energy of a structure."""
        formation_energies = []
        for atoms in structures:
            # TODO: Use cache identity stats?
            formation_energy = compute_formation_energy(atoms, self.chempot)
            formation_energies.append(formation_energy)

        return np.array(formation_energies)


if __name__ == "__main__":
    ...
