#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import abc
from typing import List, Callable

import pathlib

from ase import Atoms


class StructureGenerator(abc.ABC):

    _directory = pathlib.Path.cwd() #: Working directory.

    logger = None #: Logger instance.
    pfunc: Callable = print #: Function for outputs.

    def __init__(self, directory, *args, **kwargs):
        """"""
        self.directory = directory
        if not self.directory.exists():
            self.directory.mkdir()

        return

    @property
    def directory(self):

        return self._directory
    
    @directory.setter
    def directory(self, directory_):
        """"""
        self._directory = pathlib.Path(directory_)
        if not self._directory.exists():
            self._directory.mkdir()

        return
    
    @abc.abstractmethod
    def run(self, *args, **kwargs) -> List[Atoms]:
        """Generate structures based on rules."""
        if self.logger is not None:
            self.pfunc = self.logger.info
        self.pfunc(f"@@@{self.__class__.__name__}")

        return
    

if __name__ == "__main__":
    pass