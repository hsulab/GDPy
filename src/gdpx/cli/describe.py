import pathlib
from typing import Union

from gdpx.structures.builders.builder import StructureBuilder
from gdpx.data.array import AtomsNDArray
from gdpx.analysis.descriptors.describer import BaseDescriber
from gdpx.structures.builders.factory import canonicalise_builder
from gdpx.workflow.factory import create_describer


def describe_structures(config: dict, structures, directory: Union[str, pathlib.Path] = "./"):
    """"""
    directory = pathlib.Path(directory)

    describer = create_describer(config)
    assert isinstance(describer, BaseDescriber)

    describer.directory = directory

    # TODO: convert to a bundle of atoms?
    builder = canonicalise_builder(structures)
    assert isinstance(builder, StructureBuilder)

    frames = builder.run()  # -> List[Atoms]
    data = AtomsNDArray(frames)

    _ = describer.run(data)  # Check is_finished?

    return
