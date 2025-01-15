#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from ase import Atoms

from gdpx.group import evaluate_group_expression

from .builder import StructureModifier


class CleaveGroupModifier(StructureModifier):

    name: str = "cleave_group"

    def __init__(self, group, substrates=None, *args, **kwargs):
        """"""
        super().__init__(substrates, *args, **kwargs)

        self.group = group

        return
    
    def run(self, substrates=None, *args, **kwargs) -> list[Atoms]:
        """"""
        super().run(substrates=substrates, *args, **kwargs)

        frames = []
        for atoms in self.substrates:
            ainds = evaluate_group_expression(atoms, self.group)
            frames.append(atoms[ainds])

        return frames


if __name__ == "__main__":
    ...
