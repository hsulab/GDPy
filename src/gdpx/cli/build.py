#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import pathlib
from typing import Union

from ase.io import read, write

from ..builder.builder import StructureBuilder
from ..builder.interface import BuilderVariable


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
            # FIXME: separate reading structures from a file or a direct python object
            params = dict(method="direct", frames=params)

    builder: StructureBuilder = BuilderVariable(**params).value

    return builder


def build_structures(
    config: dict, substrates=None, size: int = 1, directory: str = "./"
):
    """"""
    directory = pathlib.Path(directory)

    builder: StructureBuilder = BuilderVariable(directory=directory, **config).value
    builder.directory = directory

    # assume substrates is a file path
    if substrates is not None:
        substrates = read(substrates, ":")

    frames = builder.run(substrates=substrates, size=size)

    write(directory / "structures.xyz", frames)

    return


if __name__ == "__main__":
    ...
