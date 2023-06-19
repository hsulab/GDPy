#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import abc
import pathlib
from typing import List, Callable

from ase import Atoms

from GDPy.core.node import AbstractNode

"""
"""


class StructureBuilder(AbstractNode):

    _directory = pathlib.Path.cwd() #: Working directory.

    logger = None #: Logger instance.
    pfunc: Callable = print #: Function for outputs.

    #: Standard print function.
    _print: Callable = print

    #: Standard debug function.
    _debug: Callable = print

    def __init__(self, substrates=None, use_tags=False, directory="./", random_seed=None, *args, **kwargs):
        """"""
        super().__init__(directory=directory, random_seed=random_seed)

        self.substrates = self._load_substrates(substrates)
        self.use_tags = use_tags

        return
    
    def _load_substrates(self, inp_sub) -> List[Atoms]:
        """"""
        substrates = inp_sub # assume it is a List of Atoms

        return substrates
    
    @abc.abstractmethod
    def run(self, *args, **kwargs) -> List[Atoms]:
        """Generate structures based on rules."""
        if self.logger is not None:
            self.pfunc = self.logger.info
        self.pfunc(f"@@@{self.__class__.__name__}")

        if not self.directory.exists():
            self.directory.mkdir(parents=True)

        return
    
StructureGenerator = StructureBuilder # compatibility

if __name__ == "__main__":
    ...