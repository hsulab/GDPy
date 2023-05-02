#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pathlib
from typing import List

from ase.io import read, write

from GDPy.core.register import registers
from GDPy.core.variable import Variable


@registers.variable.register
class DatasetVariable(Variable):

    def __init__(self, system_dirs, *args, **kwargs):
        """"""
        initial_value = self._process_dataset(system_dirs)
        super().__init__(initial_value)

        return
    
    def _process_dataset(self, system_dirs: List[str]):
        """"""
        system_dirs.sort()
        system_dirs = [pathlib.Path(s) for s in system_dirs]

        dataset = []
        for s in system_dirs:
            prefix, curr_frames = s.name, []
            xyzpaths = sorted(list(s.glob("*.xyz")))
            for p in xyzpaths:
                curr_frames.extend(read(p, ":"))
            dataset.append([prefix,curr_frames])

        return dataset

if __name__ == "__main__":
    ...