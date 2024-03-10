#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import abc
import copy
import pathlib
from typing import List, Callable

from ase import Atoms
from ase.io import read, write

from .. import config
from ..core.node import AbstractNode
from ..data.array import AtomsNDArray
from ..utils.command import dict2str


"""
"""


class StructureBuilder(AbstractNode):

    name = "builder"

    #: Standard print function.
    _print: Callable = config._print

    #: Standard debug function.
    _debug: Callable = config._debug

    def __init__(
        self, use_tags=False, directory="./", random_seed=None, *args, **kwargs
    ):
        """"""
        super().__init__(directory=directory, random_seed=random_seed)

        self.use_tags = use_tags

        return

    @abc.abstractmethod
    def run(self, substrates=None, *args, **kwargs) -> List[Atoms]:
        """Generate structures based on rules."""
        self._print(f"@@@{self.__class__.__name__}")

        self._print(f"RANDOM_SEED : {self.random_seed}")
        rng_state = self.rng.bit_generator.state
        for l in dict2str(rng_state).split("\n"):
            config._print(l)

        if not self.directory.exists():
            self.directory.mkdir(parents=True)

        return


class StructureModifier(StructureBuilder):

    name = "modifier"

    def __init__(self, substrates=None, *args, **kwargs):
        """"""
        super().__init__(*args, **kwargs)

        # TODO: substrates should also be a Builder Object
        # TODO: if substrates is a ChemiclFormula?
        if isinstance(substrates, str) or isinstance(substrates, pathlib.Path):
            substrates = pathlib.Path(substrates).absolute()
        else:
            ...
        # self._print(f"{substrates = }")

        self.substrates = self._load_substrates(substrates)
        # self._print(f"{self.substrates = }")

        return

    def _load_substrates(self, inp_sub) -> List[Atoms]:
        """"""
        substrates = None
        if isinstance(inp_sub, Atoms):
            substrates = [inp_sub]
        elif isinstance(inp_sub, list):  # assume this is a List of Atoms
            substrates = inp_sub
        elif isinstance(inp_sub, AtomsNDArray):
            substrates = inp_sub.get_marked_structures()
        else:
            # assume this is a path
            if isinstance(inp_sub, str) or isinstance(inp_sub, pathlib.Path):
                substrates = read(inp_sub, ":")
            else:
                ...

        return substrates

    def run(self, substrates=None, *args, **kwargs) -> List[Atoms]:
        """"""
        super().run(*args, **kwargs)

        # - load substrates at run
        substrates_at_run = self._load_substrates(substrates)
        if substrates_at_run is not None:
            self.substrates = substrates_at_run

        # TODO: ASE startgenerator mix builders and modifiers
        # assert self.substrates is not None, "Substrates are not set neither at inp nor at run."
        # self.substrates = [copy.deepcopy(s) for s in self.substrates]

        return


if __name__ == "__main__":
    ...