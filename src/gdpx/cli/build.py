#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import pathlib
from typing import Union

from ase.io import read, write

from gdpx.nodes.builder import BuilderVariable

from ..builder.builder import StructureBuilder


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
