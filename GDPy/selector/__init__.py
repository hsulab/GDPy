#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
import pathlib
from typing import Union, List, NoReturn

from GDPy.computation.worker.worker import AbstractWorker

from GDPy.selector.selector import AbstractSelector
from GDPy.selector.invariant import InvariantSelector
from GDPy.selector.traj import BoltzmannMinimaSelection
from GDPy.selector.descriptor import DescriptorBasedSelector
from GDPy.selector.uncertainty import DeviationSelector
from GDPy.selector.composition import ComposedSelector
from GDPy.selector.convergence import ConvergenceSelector

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
        if method == "invariant":
            selectors.append(InvariantSelector(**params))
        elif method == "convergence":
            selectors.append(ConvergenceSelector(**params))
        elif method == "boltzmann":
            selectors.append(BoltzmannMinimaSelection(**params))
        elif method == "deviation":
            selectors.append(DeviationSelector(**params, pot_worker=pot_worker))
        elif method == "descriptor":
            selectors.append(DescriptorBasedSelector(**params))
        else:
            raise RuntimeError(f"Cant find selector with method {method}.")
    
    # - try a simple composed selector
    if len(selectors) > 1:
        selector = ComposedSelector(selectors, directory=directory)
    else:
        selector = selectors[0]
        selector.directory = directory

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
    print("nframes: ", nframes)

    # -
    selected_frames = selector.select(frames)

    from ase.io import read, write
    write(directory/"selected_frames.xyz", selected_frames)

    return

if __name__ == "__main__":
    pass