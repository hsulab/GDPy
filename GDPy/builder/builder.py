#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import abc
from typing import List, Callable

import pathlib

from ase import Atoms

from GDPy.core.node import AbstractNode

"""
"""


class StructureBuilder(AbstractNode):

    _directory = pathlib.Path.cwd() #: Working directory.

    logger = None #: Logger instance.
    pfunc: Callable = print #: Function for outputs.

    def __init__(self, directory, *args, **kwargs):
        """"""
        self.directory = directory

        return
    
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
    pass