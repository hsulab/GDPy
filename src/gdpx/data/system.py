#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import re
from typing import Optional

from ase import Atoms
from ase.io import read

from gdpx.core.component import BaseComponent


class DataSystem(BaseComponent):

    """This contains a fixed-composition system.
    """

    prefix: Optional[str] = None

    _images: Optional[list[Atoms]] = None

    _tags: Optional[list[str]] = None

    def __init__(self, directory="./", pattern: str="*.xyz") -> None:
        """"""
        super().__init__(directory=directory)

        self.pattern = pattern
        self._process_dataset()

        return
    
    def _process_dataset(self):
        """"""
        wdir = self.directory
        self.prefix = wdir.name

        images = []
        xyzpaths = sorted(list(wdir.glob(self.pattern)))

        if self._images is None:
            self._images = []
        
        if self._tags is None:
            self._tags = []

        for p in xyzpaths:
            # -- read structures
            curr_frames = read(p, ":")
            n_curr_frames = len(curr_frames)
            self._debug(f"{p.name} nframes: {n_curr_frames}")
            self._images.extend(curr_frames)
            # -- add file prefix
            curr_tag = p.name.split(".")[0]
            self._tags.extend([curr_tag for _ in range(n_curr_frames)])

        return
    
    def get_matched_indices(self, pattern=r".*") -> list[Atoms]:
        """Get structures with given criteria.

        Args:
            origin: How these structures are created.
            generation: The model generatoin.

        """
        matched_indices = []
        for i, tag in enumerate(self._tags):
            if re.match(fr"{pattern}", tag) is not None:
                matched_indices.append(i)

        return matched_indices

    def __repr__(self) -> str:
        return f"DataSystem(nimages={len(self._images)})"


if __name__ == "__main__":
    ...
