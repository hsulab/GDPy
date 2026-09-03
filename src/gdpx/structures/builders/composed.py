#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from typing import Union, List

from ase import Atoms

from .builder import StructureModifier


class ComposedModifier(StructureModifier):

    """Compose several modifications into one modifier.
    """

    def __init__(self, modifiers: List[StructureModifier], substrates=None, *args, **kwargs):
        """"""
        super().__init__(substrates=substrates, *args, **kwargs)

        self.modifiers = modifiers

        return

    def run(self, substrates=None, size: Union[int,List[int]]=[1], *args, **kwargs) -> List[Atoms]:
        """"""
        super().run(substrates=substrates, *args, **kwargs)

        num_modifiers = len(self.modifiers)
        if isinstance(size, int):
            size = [size]*num_modifiers
        else:  # Must a List-like object
            ...

        num_sizes = len(size)
        if num_modifiers != num_sizes:
            raise RuntimeError(f"Inconsistent number of modifiers {num_modifiers} and target sizes {num_sizes}.")

        inp_substrates = self.substrates

        frames = []
        for i, (modifier, s) in enumerate(zip(self.modifiers, size)):
            modifier.directory = self.directory/f"chainstep.{i:02d}"
            inp_substrates = modifier.run(substrates=inp_substrates, size=s, *args, **kwargs)
        frames.extend(inp_substrates)

        return frames


if __name__ == "__main__":
    ...
  
