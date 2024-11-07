#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import re
import pathlib
from typing import List

import numpy as np

from ase import Atoms

from .builder import StructureModifier


active_iteration_pattern = re.compile(r"iter\.[0-9][0-9][0-9][0-9]")

def get_current_iteration_from_path(path: pathlib.Path) -> int:
    """"""
    parts = path.parts
    num_parts = len(parts)

    iteration = -1
    for i in range(num_parts-1, -1, -1):
        if re.match(active_iteration_pattern, parts[i]):
            iteration = int(parts[i].split(".")[-1])
            break
    else:
        ...

    return iteration

def get_previous_path_from_current_by_iteration(path: pathlib.Path) -> pathlib.Path:
    """"""
    parts = path.parts
    num_parts = len(parts)

    new_parts = list(parts)
    for i in range(num_parts-1, -1, -1):
        if re.match(active_iteration_pattern, parts[i]):
            iteration = int(parts[i].split(".")[-1])
            assert iteration > 0
            new_parts[i] = f"iter.{iteration-1:04d}"
            break
    else:
        ...

    return pathlib.Path(*new_parts)



class RouletteBuilder(StructureModifier):

    """Pick one structure randomly from several structures.
    """

    # TODO: Make this a selector?

    def __init__(self, substrates=None, use_memory: bool=False, *args, **kwargs):
        """"""
        super().__init__(substrates=substrates, *args, **kwargs)

        self.use_memory = use_memory

        return
    
    def run(self, substrates=None, size: int=1, *args, **kwargs) -> List[Atoms]:
        """"""
        super().run(substrates=substrates, *args, **kwargs)

        prev_chosen = []
        if self.use_memory:  # TODO: Check if this builder is in active?
            curr_iter = get_current_iteration_from_path(self.directory)
            assert curr_iter >= 0
            if curr_iter > 0:
                self._print(">>> Update roulette...")
                prev_wdir = get_previous_path_from_current_by_iteration(self.directory)
                prev_chosen = np.loadtxt(prev_wdir/"memory.dat").flatten()
            else:
                ...
        else:
            ...

        num_structures = len(self.substrates)

        test_chosen = self.rng.choice(num_structures, size=np.min([size*10, num_structures]), replace=False)

        curr_chosen, num_chosen = [], 0
        for i in test_chosen:
            num_chosen = len(curr_chosen)
            if num_chosen >= size:
                break
            if i not in prev_chosen:
                curr_chosen.append(i)
        assert num_chosen == size

        np.savetxt(self.directory/"memory.dat", np.hstack([prev_chosen, curr_chosen]).T)

        self._print(f"{prev_chosen =}  {curr_chosen =}")

        return [self.substrates[i] for i in curr_chosen]



if __name__ == "__main__":
    ...
  
