#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
import itertools
import pathlib
from typing import NoReturn, List, Union

import numpy as np

from ase import Atoms
from ase.io import read, write

from gdpx.core.placeholder import Placeholder
from gdpx.core.variable import Variable
from gdpx.core.operation import Operation
from gdpx.core.register import registers
from ..data.array import AtomsNDArray
from .builder import StructureBuilder

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
            "builder", method, convert_name=False, **kwargs
        )

        super().__init__(initial_value=builder, directory=directory)

        return

@registers.operation.register
class read_stru(Operation):

    def __init__(self, fname, format=None, index=":", input_nodes=[], directory="./", **kwargs) -> None:
        """"""
        super().__init__(input_nodes, directory)

        self.fname = fname
        self.format = format
        self.index = index
        self.kwargs = kwargs

        return
    
    def forward(self, *args, **kwargs) -> AtomsNDArray:
        """"""
        super().forward()
        frames = read(self.fname, format=self.format, index=self.index, **self.kwargs)
        if isinstance(frames, Atoms):
            frames = [frames] # if index is single, then read will give Atoms
        frames = AtomsNDArray(frames)

        self.status = "finished"

        return frames

@registers.operation.register
class xbuild(Operation):

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
            self._print(f"{i} - {builder.name} nframes: {len(frames)}")
            bundle.extend(frames)
        self._print(f"nframes: {len(bundle)}")

        self.status = "finished"

        return bundle

@registers.operation.register
class build(Operation):

    """Build structures without substrate structures.
    """

    def __init__(self, builder, size: int=1, directory="./") -> NoReturn:
        super().__init__(input_nodes=[builder], directory=directory)

        self.size = size

        return
    
    def forward(self, builder) -> List[Atoms]:
        """"""
        super().forward()

        builder.directory = self.directory
        curr_builder_output = self.directory/f"{builder.name}_output.xyz"
        if curr_builder_output.exists():
            frames = read(curr_builder_output, ":")
        else:
            frames = builder.run(size=self.size)
            write(curr_builder_output, frames)
        self._print(f"{builder.name} nframes: {len(frames)}")

        self.status = "finished"

        return frames

@registers.operation.register
class modify(Operation):

    def __init__(self, substrates, modifier, size: int=1, repeat: int=1, directory="./") -> None:
        """"""
        super().__init__(input_nodes=[substrates, modifier], directory=directory)

        self.size = size # create number of new structures
        self.repeat = repeat # repeat modification times for one structure

        return
    
    def forward(self, substrates: List[Atoms], modifier) -> List[Atoms]:
        """Modify inputs structures.

        A modifier only accepts one structure each time.

        """
        super().forward()

        cache_path = self.directory/f"{modifier.name}-out.xyz"
        if not cache_path.exists():
            frames = modifier.run(substrates, size=self.size)
            write(cache_path, frames)
        else:
            frames = read(cache_path, ":")

        self.status = "finished"

        return frames

def create_builder(config: Union[str, dict]) -> StructureBuilder:
    """"""
    supproted_configtypes = ["json", "yaml"]
    if isinstance(config, (str,pathlib.Path)):
        params = str(config)
        suffix = params[-4:]
        if suffix in supproted_configtypes:
            from gdpx.utils.command import parse_input_file
            params = parse_input_file(config)
        else: # assume it is an ASE readable structure file
            params = dict(
                method = "direct",
                frames = params
            )
    
    builder = BuilderVariable(**params).value

    return builder

def build_structures(config: dict, substrates=None, size: int=1, directory="./"):
    """"""
    directory = pathlib.Path(directory)

    builder = BuilderVariable(directory=directory, **config).value
    builder.directory = directory

    # assume substrates is a file path
    if substrates is not None:
        substrates = read(substrates, ":")

    frames = builder.run(substrates=substrates, size=size)

    write(directory/"structures.xyz", frames)

    return


if __name__ == "__main__":
    ...
