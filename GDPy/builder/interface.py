#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
import itertools
import pathlib
from typing import NoReturn, List, Union

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

    def __init__(self, directory="./", *args, **kwargs):
        """"""
        # - create a validator
        method = kwargs.get("method", "direct")
        builder = registers.create(
            "builder", method, convert_name=True, **kwargs
        )

        super().__init__(initial_value=builder, directory=directory)

        return

@registers.operation.register
class build(Operation):

    """Build structures without substrate structures.
    """

    def __init__(self, builders, directory="./") -> NoReturn:
        super().__init__(input_nodes=builders, directory=directory)
    
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

    def __init__(self, substrate, modifier, number: int=1, repeat: int=1, directory="./") -> NoReturn:
        """"""
        super().__init__(input_nodes=[substrate, modifier], directory=directory)

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

def create_builder(config: Union[str, dict]):
    """"""
    supproted_configtypes = ["json", "yaml"]
    if isinstance(config, (str,pathlib.Path)):
        params = str(config)
        suffix = params[-4:]
        if suffix in supproted_configtypes:
            from GDPy.utils.command import parse_input_file
            params = parse_input_file(config)
        else: # assume it is an ASE readable structure file
            params = dict(
                method = "direct",
                frames = params
            )
    
    builder = BuilderVariable(**params).value

    return builder

def build_structures(config: dict, size: int=1, directory="./"):
    """"""
    directory = pathlib.Path(directory)

    builder = BuilderVariable(directory=directory, **config).value
    frames = builder.run(size=size)

    write(directory/"structures.xyz", frames)

    return


if __name__ == "__main__":
    ...