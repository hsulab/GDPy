#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import pathlib
from typing import Union

from ..data.array import AtomsNDArray
from ..selector.interface import SelectorVariable
from ..selector.selector import AbstractSelector
from .build import create_builder


def run_selection(
    param_file: Union[str, pathlib.Path],
    structure: Union[str, dict],
    directory: Union[str, pathlib.Path] = "./",
) -> None:
    """Run selection with input selector and input structures.

    This no more accepts a worker as all data used in the selection should be
    computed in advance.

    """
    directory = pathlib.Path(directory)
    if not directory.exists():
        directory.mkdir(parents=True, exist_ok=False)

    from gdpx.utils.command import parse_input_file

    params = parse_input_file(param_file)

    selector: AbstractSelector = SelectorVariable(directory=directory, **params).value
    selector.directory = directory

    # - read structures
    builder = create_builder(structure)
    frames = builder.run()  # -> List[Atoms]

    # TODO: convert to a bundle of atoms?
    data = AtomsNDArray(frames)

    # -
    selected_frames = selector.select(data)

    from ase.io import read, write

    write(directory / "selected_frames.xyz", selected_frames)

    return


if __name__ == "__main__":
    ...
