#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
from pathlib import Path
from typing import Union, List, NoReturn

import numpy as np

from ase import Atoms
from ase.io import read, write

from GDPy.core.datatype import isAtomsFrames, isTrajectories
from GDPy.selector.selector import AbstractSelector


class ComposedSelector(AbstractSelector):
    
    """Perform several selections consecutively.
    """

    name = "composed"

    default_parameters = dict(
        selectors = []
    )

    def __init__(self, selectors: List[AbstractSelector], directory="./", *args, **kwargs):
        """"""
        super().__init__(directory=directory, *args, **kwargs)

        self.selectors = selectors

        return
    
    def _select_indices(self, frames: List[Atoms], *args, **kwargs) -> List[int]:
        """Return selected indices."""
        # - update selectors' directories
        for s in self.selectors:
            s.directory = self._directory

        # - initial index stuff
        nframes = len(frames)
        cur_index_map = list(range(nframes))
        cur_frames = frames
        
        # - run selection
        for i, node in enumerate(self.selectors):
            # - adjust name
            prev_fname = node._fname
            node.fname = str(i) + "-" + prev_fname
            node.indent = 4
            # - map indices
            #   TODO: use _select_indices instead?
            cur_indices = node.select(cur_frames, index_map=cur_index_map, ret_indices=True)
            # - create index_map for next use
            # NOTE: sub-selector does not output global indices
            cur_frames = [frames[x] for x in cur_indices]
            cur_index_map = copy.deepcopy(cur_indices)

            node.fname = prev_fname

        return cur_indices


if __name__ == "__main__":
    pass