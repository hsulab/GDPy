#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import abc
import pathlib
from typing import Optional, Union

from gdpx.core.component import BaseComponent

"""Find possible reaction pathways in given structures.
"""


class AbstractReactor(BaseComponent):
    """Base class of an arbitrary reactor.

    A valid reactor may contain the following components:
        - A builder that offers input structures
        - A worker that manages basic dynamics task (minimisation and MD)
            - driver with two calculator for PES and BIAS
            - scheduler
        - A miner that finds saddle points and MEPs
    and the results would be trajectories and pathways for further analysis.

    """

    def __init__(
        self,
        calc,
        directory: Union[str, pathlib.Path] = "./",
        random_seed: Optional[Union[int, dict]] = None,
    ) -> None:
        """"""
        super().__init__(directory=directory, random_seed=random_seed)
        self.calc = calc

        return

    def reset(self):
        """"""
        self.calc.reset()

        return

    @abc.abstractmethod
    def run(self, structures, read_cache: bool = True, *args, **kwargs):
        """"""
        if not self.directory.exists():
            self.directory.mkdir(parents=True)

        return

    @abc.abstractmethod
    def read_convergence(self, *args, **kwargs) -> bool:
        """"""
        converged = False

        return converged


if __name__ == "__main__":
    ...

