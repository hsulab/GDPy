#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import abc
import pathlib
from typing import NoReturn, Union

class Operation(abc.ABC):

    """"""

    #: Working directory for the operation.
    _directory: Union[str,pathlib.Path] = "./"

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

    @abc.abstractmethod
    def forward(self):
        """"""

        raise NotImplementedError("Operation needs forward function.")


if __name__ == "__main__":
    ...