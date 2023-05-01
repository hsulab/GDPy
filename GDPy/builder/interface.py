#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import itertools
from typing import NoReturn, List

import numpy as np

from ase import Atoms
from ase.io import read, write

from GDPy.core.placeholder import Placeholder
from GDPy.core.variable import Variable
from GDPy.core.operation import Operation
from GDPy.core.register import registers

@registers.placeholder.register
class StructurePlaceholder(Placeholder):

    name = "structure"

    def __init__(self, *args, **kwargs):
        """"""
        super().__init__()

        return

@registers.variable.register
class BuilderVariable(Variable):

    """Build structures from the scratch."""

    def __init__(self, *args, **kwargs):
        """"""
        # - create a validator
        method = kwargs.get("method", "file")
        builder = registers.create(
            "builder", method, convert_name=True, **kwargs
        )

        initial_value = builder
        super().__init__(initial_value)

        return

@registers.operation.register
class build(Operation):

    """Build structures without substrate structures.
    """

    def __init__(self, *builders) -> NoReturn:
        super().__init__(builders)
    
    def forward(self, *args, **kwargs) -> List[Atoms]:
        """"""
        super().forward()

        bundle = []
        for i, builder in enumerate(args):
            builder.directory = self.directory
            frames = builder.run()
            write(self.directory/f"{builder.name}_output-{i}.xyz", frames)
            self.pfunc(f"{i} - {builder.name} nframes: {len(frames)}")
            bundle.extend(frames)
        self.pfunc(f"nframes: {len(bundle)}")

        return bundle

@registers.operation.register
class modify(Operation):

    def __init__(self, substrate, modifier, number: int=1, repeat: int=1) -> NoReturn:
        """"""
        super().__init__([substrate, modifier])

        self.number = number # create number of new structures
        self.repeat = repeat # repeat modification times for one structure

        return
    
    def forward(self, substrates, modifier):
        """"""
        super().forward()

        cache_path = self.directory/f"{modifier.name}-out.xyz"

        if not cache_path.exists():
            frames = []
            for substrate in substrates:
                for j in range(self.number):
                    curr_atoms = substrate
                    for k in range(self.repeat):
                        curr_atoms = modifier.run([curr_atoms])[0]
                    frames.append(curr_atoms)
            write(cache_path, frames)
        else:
            frames = read(cache_path, ":")

        return frames


if __name__ == "__main__":
    ...