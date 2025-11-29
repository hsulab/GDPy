import pathlib
from typing import Union

from ase.io import write

from gdpx import config
from gdpx.builder.builder import StructureBuilder
from gdpx.data.array import AtomsNDArray
from gdpx.factory.builder import canonicalise_builder
from gdpx.nodes.selector import SelectorVariable
from gdpx.selector.selector import BaseSelector
from gdpx.utils.parser import parse_input_file


def run_selection(
    param_file: Union[str, pathlib.Path],
    structures: Union[str, dict],
    directory: Union[str, pathlib.Path] = "./",
) -> None:
    """Run selection with input selector and input structures.

    This no more accepts a worker as all data used in the selection should be
    computed in advance.

    """
    directory = pathlib.Path(directory)
    if not directory.exists():
        directory.mkdir(parents=True, exist_ok=False)

    params = parse_input_file(param_file)

    # Instantiate selector
    selector = SelectorVariable(directory=directory, **params).value
    assert isinstance(selector, BaseSelector)
    selector.directory = directory

    # Produce structures
    config._print("Producing structures for selection...")
    frames_list = []
    for structure in structures:
        builder = canonicalise_builder(structure)
        assert isinstance(builder, StructureBuilder)
        frames = builder.run()  # -> List[Atoms]
        frames_list.append(frames)

    # Convert all builders' outputs to AtomsNDArray
    data = AtomsNDArray(frames_list)
    config._print(f"  input_data_structure: {data}")

    # Run selection and dump results
    config._print("Performing selection...")
    selected_frames = selector.select(data)

    write(directory / "selected_frames.xyz", selected_frames)

    return
