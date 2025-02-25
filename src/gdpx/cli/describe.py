#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import pathlib
from typing import Union

from gdpx.factory.builder import canonicalise_builder

from gdpx.data.array import AtomsNDArray
from gdpx.nodes.describer import DescriberVariable


def describe_structures(config: dict, structures, directory: Union[str,pathlib.Path]="./"):
    """"""
    directory = pathlib.Path(directory)

    describer = DescriberVariable(**config).value
    describer.directory = directory

    # TODO: convert to a bundle of atoms?
    builder = canonicalise_builder(structures)
    frames = builder.run()  # -> List[Atoms]
    data = AtomsNDArray(frames)

    is_finished = describer.run(data)

    return


if __name__ == "__main__":
    ...
  
