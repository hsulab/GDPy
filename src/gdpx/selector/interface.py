#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
import itertools
import pathlib
from typing import Union, List, NoReturn

import omegaconf

from ase import Atoms
from ase.io import read, write

from ..core.variable import Variable
from ..core.operation import Operation
from ..core.register import registers

from ..worker.worker import AbstractWorker
from ..data.array import AtomsNDArray

from ..builder.interface import build, BuilderVariable

from .selector import AbstractSelector, load_cache
from .composition import ComposedSelector


@registers.variable.register
class SelectorVariable(Variable):

    def __init__(self, selection: Union[dict, List[dict]], directory="./", *args, **kwargs) -> None:
        """Define a Variable that has a Selector."""
        # We can define a selector in two different ways:
        # The Dict must have a selection key
        # - a Dict that defines a single selector
        # - a List of Dict that defines several selectors, 
        #   which will be converted into a composed one
        selection = copy.deepcopy(selection)
        if isinstance(selection, dict) or isinstance(selection, omegaconf.dictconfig.DictConfig):
            selection = [selection]
        elif isinstance(selection, list) or isinstance(selection, omegaconf.listconfig.ListConfig):
            ...
        else:
            raise TypeError(f"Unknown type of {selection =}.")

        selectors = []
        for params in selection:
            method = params.pop("method", None)
            selector = registers.create("selector", method, convert_name=True, **params)
            selectors.append(selector)
        nselectors = len(selectors)
        if nselectors > 1:
            selector = ComposedSelector(selectors)
        else:
            selector = selectors[0]

        super().__init__(initial_value=selector, directory=directory)

        return


@registers.operation.register
class select(Operation):

    cache_fname = "selected_frames.xyz"

    def __init__(self, structures, selector, directory="./", *args, **kwargs):
        """"""
        super().__init__(input_nodes=[structures, selector], directory=directory)

        return

    @Operation.directory.setter
    def directory(self, directory_) -> NoReturn:
        """"""
        super(select, select).directory.__set__(self, directory_)

        return

    def _preprocess_input_nodes(self, input_nodes):
        """"""
        structures, selector = input_nodes
        if isinstance(structures, str) or isinstance(structures, pathlib.Path):
            # TODO: check if it is a molecule name
            structures = build(
                BuilderVariable(
                    directory=self.directory / "structures",
                    method="reader",
                    fname=structures,
                )
            )
        # We can define a selector in two different ways:
        # The Dict must have a selection key
        # - a Dict that defines a single selector
        # - a List of Dict that defines several selectors, 
        #   which will be converted into a composed one
        if isinstance(selector, dict) or isinstance(
            selector, omegaconf.dictconfig.DictConfig
        ):
            selector = SelectorVariable(
                directory=self.directory / "selector", **selector
            )
        #self._print(f"{selector = }")

        return structures, selector

    def forward(
        self, structures: AtomsNDArray, selector: AbstractSelector
    ) -> AtomsNDArray:
        """"""
        super().forward()
        selector.directory = self.directory

        cache_fpath = self.directory / self.cache_fname
        if not cache_fpath.exists():
            new_frames = selector.select(structures)
            write(cache_fpath, new_frames)
        else:
            markers = load_cache(selector.info_fpath)
            structures.markers = markers
            if cache_fpath.stat().st_size != 0:
                new_frames = read(cache_fpath, ":")
            else: # sometimes selection gives no structures and writes empty file
                new_frames = []
        self._print(f"nframes: {len(new_frames)}")

        num_new_frames = len(new_frames)
        if num_new_frames > 0:
            self.status = "finished"
        else:
            self.status = "exit"

        return structures
    
    # NOTE: This operation exits when no structures are selected
    #       so we donot need convergence check here?
    #def report_convergence(self, *args, **kwargs) -> bool:
    #    """"""
    #    input_nodes = self.input_nodes
    #    assert self.status == "finished", f"Operation {self.directory.name} cannot report convergence without forwarding."
    #    selector = input_nodes[1].output

    #    self._print(f"{selector.__class__.__name__} Convergence")

    #    cache_fpath = self.directory / self.cache_fname # MUST EXIST
    #    if cache_fpath.stat().st_size != 0:
    #        new_frames = read(cache_fpath, ":")
    #    else: # sometimes selection gives no structures and writes empty file
    #        new_frames = []
    #    num_new_frames = len(new_frames)
    #    if num_new_frames == 0:
    #        converged = True
    #    else:
    #        converged = False

    #    return converged


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

    selector = SelectorVariable(
        directory=directory, **params
    ).value
    selector.directory = directory

    # - read structures
    from gdpx.builder import create_builder

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
