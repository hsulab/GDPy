#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
import pathlib
from typing import Union, List, NoReturn

from ase.io import read, write

from GDPy.core.variable import Variable
from GDPy.core.operation import Operation
from GDPy.core.register import registers
from GDPy.computation.worker.worker import AbstractWorker
from GDPy.selector.selector import AbstractSelector
from GDPy.selector.invariant import InvariantSelector
from GDPy.selector.composition import ComposedSelector


@registers.variable.register
class SelectorVariable(Variable):

    def __init__(self, selection):
        """"""
        selector = create_selector(selection)
        super().__init__(selector)


@registers.operation.register
class select(Operation):

    cached_fname = "selected_frames.xyz"

    def __init__(self, frames, selector):
        """"""
        super().__init__([frames,selector])

        return
    
    @Operation.directory.setter
    def directory(self, directory_) -> NoReturn:
        """"""
        super(select, select).directory.__set__(self, directory_)

        return
    
    def forward(self, frames, selector: AbstractSelector):
        """"""
        super().forward()

        selector.directory = self.directory

        cached_fpath = self.directory/self.cached_fname
        if not cached_fpath.exists():
            new_frames = selector.select(frames)
            write(cached_fpath, new_frames)
        else:
            new_frames = read(cached_fpath, ":")
        self.pfunc(f"nframes: {len(new_frames)}")

        return new_frames


def create_selector(
    input_list: List[dict], directory: Union[str,pathlib.Path]=pathlib.Path.cwd(), 
    pot_worker: AbstractWorker=None
) -> AbstractSelector:
    """Create a selector based on arguments.

    Args:
        input_list: A list of each selector's parameters.
        directory: Working directory.
        pot_worker: A worker for potential computations.
    
    Returns:
        An instance of the AbstractSelector.

    """
    selectors = []

    if not input_list:
        selectors.append(InvariantSelector())

    for s in input_list:
        params = copy.deepcopy(s)
        method = params.pop("method", None)
        # -- frame-based
        if method == "invariant":
            cur_selector = InvariantSelector(**params)
        elif method == "descriptor":
            from GDPy.selector.descriptor import DescriptorBasedSelector
            cur_selector = DescriptorBasedSelector(**params)
        elif method == "graph":
            from GDPy.selector.graph import GraphSelector
            cur_selector = GraphSelector(**params)
        elif method == "property":
            from GDPy.selector.property import PropertyBasedSelector
            cur_selector = PropertyBasedSelector(**params)
        else:
            raise RuntimeError(f"Cant find selector with method {method}.")
        # -- check if custom worker is set for selection
        from GDPy.potential.register import create_potter
        worker_config = params.pop("worker", None)
        if worker_config is not None:
            worker = create_potter(worker_config) # register calculator, and scheduler if exists
        else:
            worker = None
        cur_selector.attach_worker(worker)
        # -- 
        selectors.append(cur_selector)
    
    # - try a simple composed selector
    selector = ComposedSelector(selectors, directory=directory)

    return selector

def run_selection(
    param_file: Union[str,pathlib.Path], structure: Union[str,dict], 
    directory: Union[str,pathlib.Path]="./", potter: AbstractWorker=None
) -> NoReturn:
    """Run selection with input selector and input structures.
    """
    directory = pathlib.Path(directory)
    if not directory.exists():
        directory.mkdir(parents=True, exist_ok=False)

    from GDPy.utils.command import parse_input_file
    params = parse_input_file(param_file)

    selector = create_selector(
        params["selection"], directory=directory, pot_worker=potter
    )

   # - read structures
    from GDPy.builder import create_generator
    generator = create_generator(structure)
    frames = generator.run()
    nframes = len(frames)

    # -
    selected_frames = selector.select(frames)

    from ase.io import read, write
    write(directory/"selected_frames.xyz", selected_frames)

    return

if __name__ == "__main__":
    ...