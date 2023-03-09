#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import abc
import pathlib
from typing import NoReturn, Union

"""Every working component is represented by a node.
"""

class AbstractNode(abc.ABC):

    #: Working directory.
    _directory: pathlib.Path = "./"

    #: Standard print function.
    _print = print

    def __init__(self, directory: Union[str,pathlib.Path]="./", *args, **kwargs):
        """"""
        self.directory = directory

        return

    @property
    def directory(self) -> pathlib.Path:
        """Working directory.

        Note:
            When setting directory, some additional files are re-directed.

        """
        return self._directory
    
    @directory.setter
    def directory(self, directory_: Union[str,pathlib.Path]) -> NoReturn:
        self._directory = pathlib.Path(directory_)
        #if not self._directory.exists():
        #    self._directory.mkdir(parents=True)

        return 


def create_node(params: dict):
    """Create a node from parameters."""

    return

if __name__ == "__main__":
    ...