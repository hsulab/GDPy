#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import List

from ase import Atoms

from GDPy.builder.builder import StructureGenerator


class DirectGenerator(StructureGenerator):

    _frames = []

    def __init__(self, frames, directory="./", *args, **kwargs):
        super().__init__(directory, *args, **kwargs)
        self._frames = frames

        return
    
    @property
    def frames(self):
        return self._frames
    
    def run(self, *args, **kwargs) -> List[Atoms]:
        """"""
        return self.frames


if __name__ == "__main__":
    pass