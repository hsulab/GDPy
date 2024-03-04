#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import abc
import copy
import pathlib
from typing import NoReturn, Union, Callable

import numpy as np

from . import config

"""Every working component is represented by a node.
"""

class AbstractNode(abc.ABC):

    #: Node name.
    name: str = "node"

    #: The random seed when initialising the object.
    _init_random_seed: Union[int, dict] = None

    #: Working directory.
    _directory: pathlib.Path = "./"

    #: Standard print function.
    _print: Callable = config._print

    #: Standard debug function.
    _debug: Callable = config._debug

    #: Default parameters.
    default_parameters: dict = dict()

    def __init__(
        self, directory: Union[str,pathlib.Path]="./", 
        random_seed: Union[int, dict]=None, *args, **kwargs
    ):
        """"""
        # - set working directory
        self.directory = directory

        # - set random generator
        self._init_random_seed = random_seed
        self.set_rng(seed=random_seed)

        # - number of processors
        self.njobs = config.NJOBS

        return
    
    @property
    def init_random_seed(self):
        """The random seed at the initialisation of the object.

        This cannot be changed after the object is initialised.
        
        """

        return self._init_random_seed

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
            seed = int(config.GRNG.integers(0, 1e8))
        self.random_seed = seed

        # - assign random seeds
        if isinstance(seed, int): # TODO: numpy int?
            self.rng = np.random.Generator(np.random.PCG64(seed))
        elif isinstance(seed, dict): # TODO: omegaconf dict?
            self.rng = np.random.Generator(np.random.PCG64())
            self.rng.bit_generator.state = seed
        else:
            ...

        return


if __name__ == "__main__":
    ...