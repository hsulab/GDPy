#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pathlib
from typing import List

from ase.io import read, write

from .. import config
from gdpx.core.register import registers
from gdpx.core.variable import Variable
from .system import DataSystem


@registers.variable.register
class TempdataVariable(Variable):

    def __init__(self, system_dirs, *args, **kwargs):
        """"""
        initial_value = self._process_dataset(system_dirs)
        super().__init__(initial_value)

        return
    
    def _process_dataset(self, system_dirs: List[str]):
        """"""
        #system_dirs.sort()
        system_dirs = [pathlib.Path(s) for s in system_dirs]

        dataset = []
        for s in system_dirs:
            prefix, curr_frames = s.name, []
            xyzpaths = sorted(list(s.glob("*.xyz")))
            for p in xyzpaths:
                curr_frames.extend(read(p, ":"))
            dataset.append([prefix, curr_frames])

        return dataset


@registers.variable.register
class NamedTempdataVariable(Variable):

    def __init__(self, system_dirs, *args, **kwargs):
        """"""
        initial_value = self._process_dataset(system_dirs)
        super().__init__(initial_value)

        return
    
    def _process_dataset(self, system_dirs: List[str]):
        """"""
        #system_dirs.sort()
        system_dirs = [pathlib.Path(s) for s in system_dirs]

        data_systems = []
        for s in system_dirs:
            config._debug(str(s))
            d = DataSystem(directory=s)
            data_systems.append(d)

        return data_systems


@registers.variable.register
class DatasetVariable(Variable):

    def __init__(self, name, directory="./", *args, **kwargs):
        """"""
        dataset = registers.create("dataloader", name, convert_name=True, **kwargs)
        super().__init__(initial_value=dataset, directory=directory)

        return

if __name__ == "__main__":
    ...