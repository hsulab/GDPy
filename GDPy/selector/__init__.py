#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
import pathlib

from GDPy.selector.invariant import InvariantSelector
from GDPy.selector.traj import BoltzmannMinimaSelection
from GDPy.selector.descriptor import DescriptorBasedSelector
from GDPy.selector.uncertainty import DeviationSelector
from GDPy.selector.composition import ComposedSelector
from GDPy.selector.convergence import ConvergenceSelector

def create_selector(input_list: list, directory=pathlib.Path.cwd(), pot_worker=None):
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

def run_selection(param_file, structure, directory=pathlib.Path.cwd(), potter=None):
    """"""
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