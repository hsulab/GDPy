#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pathlib
from typing import Callable, NoReturn, Union

from .. import config

class Variable:

    """Intrinsic, changeable parameter of a graph.
    """

    identifier: str = "vx"

    #: Working directory for the operation.
    _directory: Union[str,pathlib.Path] = pathlib.Path.cwd()

    #: Working status that should be always finished.
    status = "finished"

    #: Standard print function.
    _print: Callable = config._print

    #: Standard debug function.
    _debug: Callable = config._debug

    def __init__(self, initial_value=None, directory="./"):
        """"""
        self.value = initial_value
        self.consumers = []

        self.directory = directory

        #_default_graph.variables.append(self)

        return

    @property
    def directory(self):
        """"""

        return self._directory
    
    @directory.setter
    def directory(self, directory_) -> NoReturn:
        """"""
        self._directory = pathlib.Path(directory_)

        return
    
    def reset(self):
        """Reset node's output and status."""
        if hasattr(self, "output"):
            delattr(self, "output")

        return
    
    def reset_random_seed(self, mode="init"):
        """"""
        if hasattr(self, "_reset_random_seed"):
            self._reset_random_seed(mode=mode)

        return


class DummyVariable(Variable):

    ...


if __name__ == "__main__":
    ...
