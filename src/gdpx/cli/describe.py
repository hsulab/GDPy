import pathlib
from typing import Union

from gdpx.builder.builder import StructureBuilder
from gdpx.data.array import AtomsNDArray
from gdpx.describer.describer import BaseDescriber
from gdpx.factory.builder import canonicalise_builder
from gdpx.nodes.describer import DescriberVariable


def describe_structures(config: dict, structures, directory: Union[str, pathlib.Path] = "./"):
    """"""
    directory = pathlib.Path(directory)

    describer = DescriberVariable(**config).value
    assert isinstance(describer, BaseDescriber)

    describer.directory = directory

    # TODO: convert to a bundle of atoms?
    builder = canonicalise_builder(structures)
    assert isinstance(builder, StructureBuilder)

    frames = builder.run()  # -> List[Atoms]
    data = AtomsNDArray(frames)

    _ = describer.run(data)  # Check is_finished?

    return
