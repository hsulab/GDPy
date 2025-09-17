#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import itertools
from typing import List, Union

import ase
import numpy as np
from ase import Atoms
from ase.build import molecule
from ase.collections import g2
from ase.data import atomic_numbers
from ase.formula import Formula
from ase.io import read

from gdpx.utils.strconv import string_to_integers


def convert_string_to_atoms(species: str) -> Atoms:
    """"""
    atoms = None
    if species in ase.data.chemical_symbols:
        atoms = Atoms(species, positions=[[0.0, 0.0, 0.0]])
    elif species in g2.names:
        atoms = molecule(species)
    elif species == "CHOO":
        atoms = Atoms(
            "CHOO",
            positions=[
                [8.95, 9.83, 11.71],
                [8.95, 10.03, 12.80],
                [10.08, 9.75, 11.15],
                [7.80, 9.73, 11.16],
            ],
        )
    elif species.endswith(".xyz"):
        frames = read(species, ":")  # TODO: check non-pbc molecule only?
        assert len(frames) == 1, f"Only one frame is expected in `{species}`."
        atoms = frames[0]
    else:
        raise RuntimeError(f"Cannot create species `{species}`.")

    return atoms


class ChemicalSpecies:
    """Define a structure from its chemical name or from an external structure file."""

    ...


def get_chemical_species_from_kwpairs(name: str, number: Union[int, str]):
    """"""
    species = []
    if isinstance(number, int):
        species = [(name, number)]
    elif isinstance(number, str):
        numbers = string_to_integers(number, convention="lmp", out_convention="lmp")
        for num in numbers:
            species.append((name, num))
    else:
        raise RuntimeError(f"number must be int or str but `{number}` is given.")

    return species


class CompositionSpace:

    def __init__(self, composition):
        """"""
        _compositions = []
        if isinstance(composition, dict):
            entries = []
            for k, v in composition.items():
                entries.append(get_chemical_species_from_kwpairs(name=k, number=v))
            # Sort species by name to make the composition order consistent
            # Though we will sort fragments in insert
            entries = sorted(entries, key=lambda e: e[0])
            _compositions = list(itertools.product(*entries, repeat=1))
        elif isinstance(composition, list):
            ...
        else:
            raise RuntimeError()

        # Something like [(('H', 2), ('O', 1))]
        self._compositions = _compositions
        assert len(self._compositions) > 0, f"`{_compositions}` must have at least one choice."

        return

    def get_chemical_symbols(self):
        """Get possible chemical symbols in the composition space."""
        chemical_symbols = []
        for comp in self._compositions:
            for name, _ in comp:  # (name, numb)
                if name not in chemical_symbols:
                    name_ = name.strip(".xyz")
                    chemical_symbols.extend(Formula(name_).count().keys())
        chemical_symbols = list(set(chemical_symbols))

        return chemical_symbols

    def get_chemical_numbers(self):
        """Get possible chemical numbers in the composition space."""
        chemical_symbols = self.get_chemical_symbols()
        chemical_numbers = [atomic_numbers[s] for s in chemical_symbols]

        return chemical_numbers

    def get_fragments_from_one_composition(self, rng=np.random.default_rng()) -> List[Atoms]:
        """"""
        num_compositions = len(self._compositions)
        idx = rng.choice(num_compositions, size=1, replace=False)[0]
        composition = self._compositions[idx]

        fragments = list(
            itertools.chain(
                *[[convert_string_to_atoms(name) for _ in range(number)] for (name, number) in composition]
            )
        )

        return fragments


if __name__ == "__main__":
    ...
