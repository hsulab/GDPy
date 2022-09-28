#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import List
from pathlib import Path

import numpy as np

from ase import Atoms

from GDPy.selector.selector import AbstractSelector


class GraphSelector(AbstractSelector):

    name = "graph"

    default_parameters = dict()

    def __init__(self, *args, **kwargs):
        """"""
        super().__init__(*args, **kwargs)

        return
    
    def _select_indices(self, frames: List[Atoms], *args, **kwargs) -> List[int]:
        """"""
        selected_indices = []

        raise NotImplementedError(f"{self.name} is not implemented...")


if __name__ == "__main__":
    pass