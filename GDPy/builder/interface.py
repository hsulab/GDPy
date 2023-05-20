#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
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
            curr_builder_output = self.directory/f"{builder.name}_output-{i}.xyz"
            if curr_builder_output.exists():
                frames = read(curr_builder_output, ":")
            else:
                frames = builder.run()
                write(curr_builder_output, frames)
            self.pfunc(f"{i} - {builder.name} nframes: {len(frames)}")
            bundle.extend(frames)
        self.pfunc(f"nframes: {len(bundle)}")

        self.status = "finished"

        return bundle

@registers.operation.register
class modify(Operation):

    def __init__(self, substrate, modifier, number: int=1, repeat: int=1) -> NoReturn:
        """"""
        super().__init__([substrate, modifier])

        self.number = number # create number of new structures
        #self.repeat = repeat # repeat modification times for one structure

        return
    
    def forward(self, substrates, modifier):
        """Modify inputs structures.

        A modifier only accepts one structure each time.

        """
        super().forward()

        cache_path = self.directory/f"{modifier.name}-out.xyz"

        if not cache_path.exists():
            frames = []
            for substrate in substrates:
                curr_atoms = copy.deepcopy(substrate)
                curr_frames = modifier.run(curr_atoms, size=self.number)
                frames.extend(curr_frames)
            write(cache_path, frames)
        else:
            frames = read(cache_path, ":")

        self.status = "finished"

        return frames


if __name__ == "__main__":
    ...