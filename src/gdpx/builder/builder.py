#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import abc
import pathlib
from typing import Optional, Union

from ase import Atoms
from ase.io import read

from gdpx.core.component import BaseComponent

from ..data.array import AtomsNDArray
from ..utils.command import dict2str


class StructureBuilder(BaseComponent):

    name = "builder"

    def __init__(
        self,
        use_tags: bool = False,
        directory="./",
        random_seed: Optional[Union[int, dict]] = None,
    ):
        """"""
        super().__init__(directory=directory, random_seed=random_seed)

        self.use_tags = use_tags

        return

    @abc.abstractmethod
    def run(self, substrates=None, *args, **kwargs) -> list[Atoms]:
        """Generate structures based on rules."""
        if self.__class__.__name__ != "ComposedModifier":
            self._print(f"-->{self.__class__.__name__}")
            self._print(f"RANDOM_SEED : {self.random_seed}")
            rng_state = self.rng.bit_generator.state
            for l in dict2str(rng_state).split("\n"):  # type: ignore
                self._print(l)
        else:
            ...

        if not self.directory.exists():
            self.directory.mkdir(parents=True)

        ...


class StructureModifier(StructureBuilder):

    name = "modifier"

    def __init__(self, substrates=None, *args, **kwargs):
        """"""
        super().__init__(*args, **kwargs)

        # TODO: substrates should also be a Builder Object
        # TODO: if substrates is a ChemiclFormula?
        self._input_substrates = None
        if isinstance(substrates, str) or isinstance(substrates, pathlib.Path):
            substrates = pathlib.Path(substrates).resolve()
            self._input_substrates = str(substrates)
        else:
            ...

        self.substrates = self._canonicalise_substrates(substrates)

        return

    def _canonicalise_substrates(self, inp_sub) -> list[Atoms]:
        """Convert input substrates to a list of Atoms."""
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

        return substrates  # type: ignore

    @abc.abstractmethod
    def run(self, substrates=None, *args, **kwargs) -> list[Atoms]:
        """Generate structures based on rules."""
        super().run(*args, **kwargs)

        # Load substrates at run time
        substrates_at_run = self._canonicalise_substrates(substrates)
        if substrates_at_run is not None:
            self.substrates = substrates_at_run

        # TODO: ASE startgenerator mix builders and modifiers
        # assert self.substrates is not None, "Substrates are not set neither at inp nor at run."
        # self.substrates = [copy.deepcopy(s) for s in self.substrates]

        ...


if __name__ == "__main__":
    ...
