#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import abc
import pathlib
from typing import Any, Callable, Optional, Union

import numpy as np
import omegaconf

from . import config

"""Every working component is represented by a node.
"""


class BaseComponent(abc.ABC):

    #: Node name.
    name: str = "component"

    #: Standard print function.
    _print: Callable = config._print

    #: Standard debug function.
    _debug: Callable = config._debug

    #: Default parameters.
    default_parameters: dict = dict()

    def __init__(
        self,
        directory: Union[str, pathlib.Path] = "./",
        random_seed: Optional[Union[int, dict]] = None,
    ):
        """"""
        # Set the working directory.
        self._directory = pathlib.Path(directory)

        # Set the random number generator
        self._init_random_seed = random_seed

        self.set_rng(seed=random_seed)

        # Set number of processors
        self.njobs = config.NJOBS

        return

    @property
    def init_random_seed(self) -> Optional[Union[int, dict]]:
        """The random seed at the initialisation of the object.

        This cannot be changed after the object is initialised.

        """

        return self._init_random_seed

    @property
    def directory(self) -> pathlib.Path:
        """Working directory.

        Note:
            When setting directory, some additional files are re-directed.

        Returns:
            The working directory path.

        """

        return self._directory

    @directory.setter
    def directory(self, directory: Union[str, pathlib.Path]) -> None:
        """"""
        self._directory = pathlib.Path(directory)

        return

    def set_rng(self, seed: Any = None) -> None:
        """Set the random number generator.

        We must use self.rng to generate any random things for reproducibility.

        """
        # Get an integer from the global rng if seed is not set,
        # and store the integer as the seed.
        if seed is None:
            seed = int(config.GRNG.integers(0, 1e8))  # type: ignore

        self.random_seed = seed

        # Set up random seeds
        if isinstance(seed, int) or isinstance(seed, np.integer):
            self.rng = np.random.Generator(np.random.PCG64(seed))
        elif isinstance(seed, dict) or isinstance(seed, omegaconf.DictConfig):
            self.rng = np.random.Generator(np.random.PCG64())
            self.rng.bit_generator.state = seed
        else:
            raise Exception(f"Unknown random seed `{seed}` type `{type(seed)}`")

        return


# For backward compatibility
AbstractNode = BaseComponent


if __name__ == "__main__":
    ...
