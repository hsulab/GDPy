#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import List, Union

from ase import Atoms

from .builder import StructureModifier


class RepeatModifier(StructureModifier):

    name = "repeat"

    def __init__(self, repeat: Union[int,List[int]]=1, substrates=None, *args, **kwargs):
        """"""
        super().__init__(substrates, *args, **kwargs)

        self.repeat = repeat

        return
    
    def run(self, substrates: List[Atoms], size: int=1, *args, **kwargs):
        """"""
        super().run(substrates=substrates, *args, **kwargs)

        frames = []
        for substrate in self.substrates:
            frames.append(substrate.repeat(self.repeat))

        return frames


if __name__ == "__main__":
    ...