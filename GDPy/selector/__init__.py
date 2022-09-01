#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
import pathlib

from GDPy.selector.abstract import ConvergenceSelector, ComposedSelector
from GDPy.selector.traj import BoltzmannMinimaSelection
from GDPy.selector.descriptor import DescriptorBasedSelector
from GDPy.selector.uncertainty import DeviationSelector

def create_selector(input_list: list, directory=pathlib.Path.cwd()):
    selectors = []
    for s in input_list:
        params = copy.deepcopy(s)
        method = params.pop("method", None)
        if method == "convergence":
            selectors.append(ConvergenceSelector(**params))
        elif method == "deviation":
            selectors.append(DeviationSelector(**params))
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

if __name__ == "__main__":
    pass