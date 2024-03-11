#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
import collections
import itertools
import pathlib

from typing import NoReturn, List, Union

import omegaconf

import numpy as np

from ase import Atoms
from ase.io import read, write

from ..core.placeholder import Placeholder
from ..core.variable import Variable
from ..core.operation import Operation
from ..core.register import registers
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
        builder = registers.create("builder", method, convert_name=False, **kwargs)

        super().__init__(initial_value=builder, directory=directory)

        return

    @Variable.directory.setter
    def directory(self, directory_) -> NoReturn:
        """"""
        self._directory = pathlib.Path(directory_)

        # Value is the attached builder
        self.value.directory = self._directory

        return

    def _reset_random_seed(self, mode: str = "init"):
        """Rewind random state to the one at the initialisation of the object."""
        if mode == "init":
            self.value.set_rng(seed=self.value.init_random_seed)
        elif mode == "zero":
            self.value.set_rng(seed=None)
        else:
            raise RuntimeError(f"INCORRECT RESET RANDOM SEED MODE {mode}.")

        return


@registers.operation.register
class read_stru(Operation):

    def __init__(
        self, fname, format=None, index=":", input_nodes=[], directory="./", **kwargs
    ) -> None:
        """"""
        super().__init__(input_nodes, directory)

        self.fname = fname  # This is broadcastable...
        self.format = format
        self.index = index
        self.kwargs = kwargs

        return

    def forward(self, *args, **kwargs) -> AtomsNDArray:
        """"""
        super().forward()

        # - check params
        if isinstance(self.fname, str):
            fname_ = [self.fname]
        else:  # assume it is an iterable object
            fname_ = self.fname

        # - read structures
        frames = []
        for curr_fname in fname_:
            self._print(f"read {curr_fname}")
            curr_frames = read(
                curr_fname, format=self.format, index=self.index, **self.kwargs
            )
            if isinstance(curr_frames, Atoms):
                curr_frames = [
                    curr_frames
                ]  # if index is single, then read will give Atoms
            frames.extend(curr_frames)

        frames = AtomsNDArray(frames)
        self._print(f"shape of structures: {frames.shape}")

        self.status = "finished"

        return frames


@registers.operation.register
class write_stru(Operation):

    def __init__(
        self, fname, structures, format="extxyz", directory="./", *args, **kwargs
    ) -> None:
        """"""
        input_nodes = [structures]
        super().__init__(input_nodes, directory)

        self.fname = fname
        self.format = format
        self.kwargs = kwargs

        return

    def forward(self, structures, *args, **kwargs):
        """"""
        super().forward()

        write(self.directory / self.fname, structures, format=self.format)

        self.status = "finished"

        return


@registers.operation.register
class xbuild(Operation):
    """Build structures without substrate structures."""

    def __init__(self, builders, directory="./") -> NoReturn:
        super().__init__(input_nodes=builders, directory=directory)

    def forward(self, *args, **kwargs) -> List[Atoms]:
        """"""
        super().forward()

        bundle = []
        for i, builder in enumerate(args):
            builder.directory = self.directory
            curr_builder_output = self.directory / f"{builder.name}_output-{i}.xyz"
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
    """Build structures without substrate structures."""

    def __init__(self, builder, size: int = 1, directory="./") -> NoReturn:
        super().__init__(input_nodes=[builder], directory=directory)

        self.size = size

        return

    def forward(self, builder) -> List[Atoms]:
        """"""
        super().forward()

        builder.directory = self.directory
        curr_builder_output = self.directory / f"{builder.name}_output.xyz"
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

    def __init__(
        self, substrates, modifier, size: int = 1, repeat: int = 1, directory="./"
    ) -> None:
        """"""
        super().__init__(input_nodes=[substrates, modifier], directory=directory)

        self.size = size  # create number of new structures
        self.repeat = repeat  # repeat modification times for one structure

        return

    @Operation.directory.setter
    def directory(self, directory_) -> None:
        """"""
        self._directory = pathlib.Path(directory_)

        self.input_nodes[0].directory = self._directory / "substrates"
        self.input_nodes[1].directory = self._directory / "modifier"

        return

    def _preprocess_input_nodes(self, input_nodes):
        """"""
        substrates, modifier = input_nodes
        if isinstance(substrates, str) or isinstance(substrates, pathlib.Path):
            # TODO: check if it is a molecule name
            substrates = build(
                BuilderVariable(
                    directory=self.directory / "substrates",
                    method="reader",
                    fname=substrates,
                )
            )
        if isinstance(modifier, dict) or isinstance(
            modifier, omegaconf.dictconfig.DictConfig
        ):
            modifier = BuilderVariable(
                directory=self.directory / "modifier", **modifier
            )

        return substrates, modifier

    def forward(self, substrates: List[Atoms], modifier) -> List[Atoms]:
        """Modify inputs structures.

        A modifier only accepts one structure each time.

        """
        super().forward()

        cache_path = self.directory / f"{modifier.name}-out.xyz"
        if not cache_path.exists():
            frames = modifier.run(substrates, size=self.size)
            write(cache_path, frames)
        else:
            frames = read(cache_path, ":")

        self.status = "finished"

        return frames


def create_builder(config: Union[str, dict]) -> StructureBuilder:
    """"""
    supported_configtypes = ["json", "yaml"]
    if isinstance(config, (str, pathlib.Path)):
        params = str(config)
        suffix = params[-4:]
        if suffix in supported_configtypes:
            from gdpx.utils.command import parse_input_file

            params = parse_input_file(config)
        else:  # assume it is an ASE readable structure file
            params = dict(method="direct", frames=params)

    builder = BuilderVariable(**params).value

    return builder


def build_structures(config: dict, substrates=None, size: int = 1, directory="./"):
    """"""
    directory = pathlib.Path(directory)

    builder = BuilderVariable(directory=directory, **config).value
    builder.directory = directory

    # assume substrates is a file path
    if substrates is not None:
        substrates = read(substrates, ":")

    frames = builder.run(substrates=substrates, size=size)

    write(directory / "structures.xyz", frames)

    return


if __name__ == "__main__":
    ...
