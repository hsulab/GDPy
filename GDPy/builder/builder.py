#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import abc
from typing import List

import pathlib

from ase import Atoms


class StructureGenerator(abc.ABC):

    _directory = pathlib.Path.cwd()

    def __init__(self, directory, *args, **kwargs):
        """"""
        self.directory = directory

        return

    @property
    def directory(self):

        return self._directory
    
    @directory.setter
    def directory(self, directory_):
        """"""
        self._directory = pathlib.Path(directory_)

        return
    
    @abc.abstractmethod
    def run(self, *args, **kwargs) -> List[Atoms]:

        return
    

if __name__ == "__main__":
    pass