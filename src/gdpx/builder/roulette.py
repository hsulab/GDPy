#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from typing import List

import numpy as np

from ase import Atoms

from .builder import StructureModifier


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
            curr_iter = int(self.directory.parent.name.split(".")[-1])
            if curr_iter > 0:
                self._print(">>> Update roulette...")
                prev_wdir = (
                    self.directory.parent.parent
                    / f"iter.{str(curr_iter-1).zfill(4)}"
                    / self.directory.name
                )
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
  
