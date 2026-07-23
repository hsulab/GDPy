import copy
import pathlib
from typing import Mapping, Union

import omegaconf
from ase.io import read, write

from gdpx.core.register import registers
from gdpx.factory.components import create_selector
from gdpx.data.array import AtomsNDArray
from gdpx.nodes.builder import BuilderVariable, build
from gdpx.selector.composition import ComposedSelector
from gdpx.selector.selector import BaseSelector, load_cache
from gdpx.session.operation import Operation
from gdpx.session.variable import Variable


@registers.variable.register
class SelectorVariable(Variable):
    """A Variable that holds a Selector."""

    def __init__(
        self,
        selection: Union[dict, list[dict]],
        directory: Union[str, pathlib.Path] = "./",
    ) -> None:
        """Define a Variable that has a Selector."""
        # We can define a selector in two different ways:
        # The Dict must have a selection key
        # - a Dict that defines a single selector
        # - a list of Dict that defines several selectors,
        #   which will be converted into a composed one
        selection = copy.deepcopy(selection)
        if isinstance(selection, dict) or isinstance(selection, omegaconf.dictconfig.DictConfig):
            selection_definitions = [selection]
        elif isinstance(selection, list) or isinstance(selection, omegaconf.listconfig.ListConfig):
            selection_definitions = selection
        else:
            raise TypeError(f"Unknown type of {selection =}.")

        selectors = []
        for params in selection_definitions:
            # Check params type
            assert isinstance(params, Mapping), f"Selector definition must be a Dict, got {type(params)}."
            if isinstance(params, dict):
                ...
            elif isinstance(params, omegaconf.dictconfig.DictConfig):
                params = omegaconf.OmegaConf.to_container(params, resolve=True)
            else:
                raise TypeError(f"Unknown type of {params =}.")
            assert isinstance(params, dict), f"Selector definition must be a Dict, got {type(params)}."
            method = params.pop("method", None)
            # Instantiate selector
            selector = create_selector(dict(method=method, **params))
            selectors.append(selector)

        # Compose selectors if there are multiple ones
        num_selectors = len(selectors)
        if num_selectors > 1:
            selector = ComposedSelector(selectors)
        else:
            selector = selectors[0]

        super().__init__(initial_value=selector, directory=directory)

        return


@registers.operation.register
class select(Operation):
    cache_fname = "selected_frames.xyz"

    def __init__(
        self,
        structures,
        selector: BaseSelector,
        ignore_previous_selections: bool = False,
        directory: Union[str, pathlib.Path] = "./",
    ):
        """"""
        super().__init__(input_nodes=[structures, selector], directory=directory)

        self.ignore_previous_selections = ignore_previous_selections

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
        # - a list of Dict that defines several selectors,
        #   which will be converted into a composed one
        if isinstance(selector, dict) or isinstance(selector, omegaconf.dictconfig.DictConfig):
            selector = SelectorVariable(directory=self.directory / "selector", **selector)
        # self._print(f"{selector = }")

        return structures, selector

    def forward(self, structures: AtomsNDArray, selector: BaseSelector) -> AtomsNDArray:
        """"""
        super().forward()
        selector.directory = self.directory

        structures = AtomsNDArray(structures)
        self._print(f"{structures = }")

        # Sometimes we perform selections in parallel, thus,
        # we can ignore previous selections (markers).
        if self.ignore_previous_selections:
            structures.reset_markers()

        num_valid_structures = len(structures.get_marked_structures())
        self._print(f"-> num_valid_structures: {num_valid_structures}")

        cache_fpath = self.directory / self.cache_fname
        if not cache_fpath.exists():
            new_frames = selector.select(structures)
            write(cache_fpath, new_frames)
        else:
            markers = load_cache(selector.info_fpath)
            structures.markers = markers
            if cache_fpath.stat().st_size != 0:
                new_frames = read(cache_fpath, ":")
            else:  # sometimes selection gives no structures and writes empty file
                new_frames = []
        self._print(f"-> num_selected_structures: {len(new_frames)}")

        num_new_frames = len(new_frames)
        if num_new_frames > 0:
            self.status = "finished"
        else:
            self.status = "exit"

        return structures

    # NOTE: This operation exits when no structures are selected
    #       so we donot need convergence check here?
    # def report_convergence(self, *args, **kwargs) -> bool:
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
