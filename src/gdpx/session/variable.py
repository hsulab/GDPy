#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pathlib
from typing import Callable, NoReturn, Union

from .. import config


class Variable:
    """Intrinsic, changeable parameter of a graph."""

    #: Node ID.
    identifier: str = "vx"

    #: Working directory for the operation.
    _directory: pathlib.Path = pathlib.Path.cwd()

    #: Working status that should be always finished.
    status = "finished"

    #: Standard print function.
    _print: Callable = config._print

    #: Standard debug function.
    _debug: Callable = config._debug

    def __init__(self, initial_value=None, directory: Union[str, pathlib.Path] = "./"):
        """"""
        self._value = initial_value
        self.consumers = []

        # We need _directory as a class attribute since some variables may access the 
        # directory before the __init__ method is called.
        self.directory = directory

        return

    @property
    def value(self):
        """"""

        return self._value

    @property
    def directory(self) -> pathlib.Path:
        """"""

        return self._directory

    @directory.setter
    def directory(self, directory) -> None:
        """"""
        self._directory = pathlib.Path(directory)

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


class DummyVariable(Variable): ...


if __name__ == "__main__":
    ...
