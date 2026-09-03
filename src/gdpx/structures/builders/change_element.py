#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
import dataclasses
import itertools
from typing import Optional, List, Dict

from ase import Atoms

from .builder import StructureModifier


def get_compositions(atoms: Atoms) -> Dict[str, List[int]]:
    """"""
    compositions = {}
    for k, v in itertools.groupby(enumerate(atoms.get_chemical_symbols()), key=lambda x: x[1]):
        indices = [x[0] for x in v]
        if k in compositions:
            compositions[k].extend(indices)
        else:
            compositions[k] = indices

    return compositions


@dataclasses.dataclass
class ElementReplacement:

    #: The source element type.
    src: str = "X"

    #: The destinated element type.
    dst: str = "X"

    #: The number of atoms to exchange.
    num: int = 0

@dataclasses.dataclass
class ElementRemoval:

    #: The source element type.
    src: str = "X"

    #: The number of atoms to exchange.
    num: int = 0



class ReplaceElementModifier(StructureModifier):

    """Replace several atoms of one type to another.
    """

    def __init__(self, replacement, substrates=None, *args, **kwargs):
        """"""
        super().__init__(substrates=substrates, *args, **kwargs)

        self.elem_repl = ElementReplacement(*replacement)

        return

    def run(self, substrates=None, size: int=1, *args, **kwargs) -> List[Atoms]:
        """"""
        super().run(substrates=substrates, *args, **kwargs)

        frames = []
        for substrate in self.substrates:
            curr_frames = self._irun(substrate=substrate, size=size, *args, **kwargs)
            frames.extend(curr_frames)

        return frames
    
    def _irun(self, substrate: Atoms, size: int, *args, **kwargs) -> List[Atoms]:
        """"""
        new_atoms = copy.deepcopy(substrate)
        compositions = get_compositions(new_atoms)

        er = self.elem_repl

        num_src = len(compositions[er.src])
        if er.num < num_src:
            selected_indices = self.rng.choice(compositions[er.src], er.num, replace=False)
        else:
            selected_indices = compositions[er.src]

        num_selected = len(selected_indices)
        assert num_selected > 0

        for i in selected_indices:
            new_atoms[i].symbol = er.dst  # type: ignore

        return [new_atoms]


class RemoveElementModifier(StructureModifier):

    """Replace several atoms of one type to another.
    """

    def __init__(self, removal, substrates=None, *args, **kwargs):
        """"""
        super().__init__(substrates=substrates, *args, **kwargs)

        self.elem_remv = ElementRemoval(*removal)

        return

    def run(self, substrates=None, size: int=1, *args, **kwargs) -> List[Atoms]:
        """"""
        super().run(substrates=substrates, *args, **kwargs)

        frames = []
        for substrate in self.substrates:
            curr_frames = self._irun(substrate=substrate, size=size, *args, **kwargs)
            frames.extend(curr_frames)

        return frames
    
    def _irun(self, substrate: Atoms, size: int, *args, **kwargs) -> List[Atoms]:
        """"""
        new_atoms = copy.deepcopy(substrate)
        compositions = get_compositions(new_atoms)

        er = self.elem_remv

        num_src = len(compositions[er.src])
        if er.num < num_src:
            selected_indices = self.rng.choice(compositions[er.src], er.num, replace=False)
        else:
            selected_indices = compositions[er.src]

        num_selected = len(selected_indices)
        assert num_selected > 0

        #: del does not remove atoms.arrays, for example, forces
        del new_atoms[selected_indices]
        new_atoms.calc = None

        return [new_atoms]


if __name__ == "__main__":
    ...
  
