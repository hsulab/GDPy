#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import List
from pathlib import Path
import copy

import numpy as np

from ase import Atoms

from GDPy.selector.abstract import AbstractSelector


class InvariantSelector(AbstractSelector):

    name = "invariant"

    default_parameters = dict()

    def __init__(self, *args, **kwargs):
        """"""
        super().__init__(*args, **kwargs)

        return
    
    def select(self, frames, index_map=None, ret_indices: bool=False, *args, **kargs) -> List[Atoms]:
        """"""
        super().select(*args, **kargs)

        selected_indices = list(range(len(frames)))

        # map selected indices
        if index_map is not None:
            selected_indices = [index_map[s] for s in selected_indices]
        
        if not ret_indices:
            selected_frames = [frames[i] for i in selected_indices]
            return selected_frames
        else:
            return selected_indices


if __name__ == "__main__":
    pass