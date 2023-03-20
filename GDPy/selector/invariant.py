#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
from typing import List
from pathlib import Path

import numpy as np

from ase import Atoms

from GDPy.selector.selector import AbstractSelector


class InvariantSelector(AbstractSelector):

    """Perform an invariant selection.
    """

    name = "invariant"

    default_parameters = dict()

    def __init__(self, *args, **kwargs):
        """"""
        super().__init__(*args, **kwargs)

        return
    
    def _select_indices(self, frames: List[Atoms], *args, **kwargs) -> List[int]:
        """Return selected indices."""
        selected_indices = list(range(len(frames)))

        return selected_indices


if __name__ == "__main__":
    pass