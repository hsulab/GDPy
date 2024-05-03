#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from typing import List

from ase import Atoms

from .builder import StructureModifier
from .group import create_a_group

""""""


class CleaveGroupModifier(StructureModifier):

    name: str = "cleave_group"

    def __init__(self, group, substrates=None, *args, **kwargs):
        """"""
        super().__init__(substrates, *args, **kwargs)

        self.group = group

        return
    
    def run(self, substrates=None, *args, **kwargs) -> List[Atoms]:
        """"""
        super().run(substrates=substrates, *args, **kwargs)

        frames = []
        for atoms in substrates:
            ainds = create_a_group(atoms, self.group)
            frames.append(atoms[ainds])

        return frames


if __name__ == "__main__":
    ...
