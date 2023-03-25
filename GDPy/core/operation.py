#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import abc
import logging
import pathlib
from typing import NoReturn, Union

class Operation(abc.ABC):

    """"""

    #: Working directory for the operation.
    _directory: Union[str,pathlib.Path] = "./"

    #: Whether re-compute this operation.
    restart: bool = False

    def __init__(self, input_nodes=[]) -> NoReturn:
        """"""
        self.input_nodes = input_nodes

        # Initialize list of consumers (i.e. nodes that receive this operation's output as input)
        self.consumers = []

        # Append this operation to the list of consumers of all input nodes
        for input_node in input_nodes:
            input_node.consumers.append(self)

        # Append this operation to the list of operations in the currently active default graph
        #_default_graph.operations.append(self)

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

    @abc.abstractmethod
    def forward(self):
        """"""
        if not self.directory.exists():
            self.directory.mkdir(parents=True)
        
        self._init_logger()
        if hasattr(self, "logger") is not None:
            self.pfunc = self.logger.info
        self.pfunc(f"@@@{self.__class__.__name__}")


        return


if __name__ == "__main__":
    ...