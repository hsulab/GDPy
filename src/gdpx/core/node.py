#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import abc
import copy
import pathlib
from typing import NoReturn, Union, Callable

import numpy as np

from gdpx import config

"""Every working component is represented by a node.
"""

class AbstractNode(abc.ABC):

    #: Node name.
    name: str = "node"

    #: Working directory.
    _directory: pathlib.Path = "./"

    #: Standard print function.
    _print: Callable = config._print

    #: Standard debug function.
    _debug: Callable = config._debug

    #: Default parameters.
    default_parameters: dict = dict()

    def __init__(self, directory: Union[str,pathlib.Path]="./", random_seed: Union[int, dict]=None, *args, **kwargs):
        """"""
        # - set working directory
        self.directory = directory

        # - set random generator
        self.set_rng(seed=random_seed)

        # - number of processors
        self.njobs = config.NJOBS

        # - update parameters from kwargs
        self.parameters = copy.deepcopy(self.default_parameters)
        for k in self.parameters:
            if k in kwargs.keys():
                self.parameters[k] = kwargs[k]

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

    def set_rng(self, seed=None):
        """"""
        # - save random seed
        if seed is None:
            seed = np.random.randint(0, 1e8)
        self.random_seed = seed

        # - assign random seeds
        if isinstance(seed, int):
            self.rng = np.random.Generator(np.random.PCG64(seed))
        elif isinstance(seed, dict):
            self.rng = np.random.Generator(np.random.PCG64())
            self.rng.bit_generator.state = seed
        else:
            ...

        return

    def set(self, *args, **kwargs):
        """Set parameters."""
        for k, v in kwargs.items():
            if k in self.parameters:
                self.parameters[k] = v

        return

    def __getattr__(self, key):
        """Corresponding getattribute-function."""
        if key != "parameters" and key in self.parameters:
            return self.parameters[key]

        return object.__getattribute__(self, key)

    def as_dict(self) -> dict:
        """Return a dict that stores the current state of this node."""
        params = dict(
            name = self.__class__.__name__
        )
        params.update(**copy.deepcopy(self.parameters))

        return params


if __name__ == "__main__":
    ...