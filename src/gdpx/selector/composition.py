#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
from pathlib import Path
from typing import Union, List, NoReturn

import numpy as np

from ase import Atoms
from ase.io import read, write

from ..data.array import AtomsNDArray
from .selector import AbstractSelector


class ComposedSelector(AbstractSelector):
    """Perform several selections consecutively."""

    name = "composed"

    default_parameters = dict(selectors=[])

    def __init__(
        self, selectors: List[AbstractSelector], directory="./", *args, **kwargs
    ):
        """"""
        super().__init__(directory=directory, *args, **kwargs)

        self.selectors = selectors

        return

    def _mark_structures(self, frames: AtomsNDArray, *args, **kwargs) -> None:
        """Return selected indices."""
        # - update selectors' directories
        for s in self.selectors:
            s.directory = self._directory

        # - initial index stuff
        curr_frames = frames

        # - run selection
        for i, node in enumerate(self.selectors):
            # - adjust name
            prev_fname = node._fname
            node.fname = str(i) + "-" + prev_fname
            # - map indices
            #   TODO: use _select_indices instead?
            node.select(curr_frames)

            node.fname = prev_fname

        return


if __name__ == "__main__":
    ...
