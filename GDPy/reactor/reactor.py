#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import abc
import pathlib
import logging

from ase import Atoms

from GDPy.core.node import AbstractNode
from GDPy.builder.builder import StructureBuilder


"""Find possible reaction pathways in given structures.
"""


class AbstractReactor(AbstractNode):

    _directory = "./"

    """Base class of an arbitrary reactor.

    A valid reactor may contain the following components: 
        - A builder that offers input structures
        - A worker that manages basic dynamics task (minimisation and MD)
            - driver with two calculator for PES and BIAS
            - scheduler
        - A miner that finds saddle points and MEPs
    and the results would be trajectories and pathways for further analysis.

    """

    def __init__(self, directory="./") -> None:
        """"""
        self.directory = directory

        return

    def _init_logger(self):
        """"""
        self.logger = logging.getLogger(__name__)

        log_level = logging.INFO

        self.logger.setLevel(log_level)

        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        # - stream
        ch = logging.StreamHandler()
        ch.setLevel(log_level)
        #ch.setFormatter(formatter)

        # -- avoid duplicate stream handlers
        for handler in self.logger.handlers:
            if isinstance(handler, logging.StreamHandler):
                break
        else:
            self.logger.addHandler(ch)

        # - file
        log_fpath = self.directory/(self.__class__.__name__+".out")
        if log_fpath.exists():
            fh = logging.FileHandler(filename=log_fpath, mode="a")
        else:
            fh = logging.FileHandler(filename=log_fpath, mode="w")
        fh.setLevel(log_level)
        self.logger.addHandler(fh)
        #fh.setFormatter(formatter)

        return

    def run(self, builder: StructureBuilder, worker):
        """"""
        if self.logger is not None:
            self._print = self.logger.info
        self._print(f"@@@{self.__class__.__name__}")

        if not self.directory.exists():
            self.directory.mkdir(parents=True)
        
        return 
    
    def irun(self, atoms: Atoms, worker):
        """"""

        return


if __name__ == "__main__":
    ...