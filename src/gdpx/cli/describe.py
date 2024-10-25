#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import pathlib
from typing import Union

from ..data.array import AtomsNDArray
from ..describer.interface import DescriberVariable
from .build import create_builder


def describe_structures(config: dict, structures, directory: Union[str,pathlib.Path]="./"):
    """"""
    directory = pathlib.Path(directory)

    describer = DescriberVariable(**config).value
    describer.directory = directory

    # TODO: convert to a bundle of atoms?
    builder = create_builder(structures)
    frames = builder.run()  # -> List[Atoms]
    data = AtomsNDArray(frames)

    is_finished = describer.run(data)

    return


if __name__ == "__main__":
    ...
  
