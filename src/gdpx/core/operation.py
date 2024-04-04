#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import abc
import pathlib
from typing import Union, Callable

from .. import config


class Operation(abc.ABC):
    """"""

    identifier: str = "op"

    #: Working directory for the operation.
    _directory: Union[str, pathlib.Path] = pathlib.Path.cwd()

    #: Whether re-compute this operation
    status: str = "unfinished"  # ["unfinished", "ready", "wait", "finished"]

    #: Standard print function.
    _print: Callable = config._print

    #: Standard debug function.
    _debug: Callable = config._debug

    def __init__(self, input_nodes=[], directory: Union[str, pathlib.Path] = "./") -> None:
        """"""
        if hasattr(self, "_preprocess_input_nodes"):
            self.input_nodes = self._preprocess_input_nodes(input_nodes)
        else:
            self.input_nodes = input_nodes

        self.directory = directory

        # Initialize list of consumers
        # (i.e. nodes that receive this operation's output as input)
        self.consumers = []

        # Append this operation to the list of consumers of all input nodes
        for input_node in self.input_nodes:
            input_node.consumers.append(self)

        return

    @property
    def directory(self) -> pathlib.Path:
        """"""

        return self._directory

    @directory.setter
    def directory(self, directory_) -> None:
        """"""
        self._directory = pathlib.Path(directory_)

        return

    def reset(self):
        """Reset node's output and status."""
        if hasattr(self, "output"):
            delattr(self, "output")
            self.status = "unfinished"

        return
    
    def is_about_to_exit(self) -> bool:
        """Check whether this operation has an input is about to exit."""
        status = [node.status == "exit" for node in self.input_nodes]
        if any(status):
            return True
        else:
            return False

    def is_ready_to_forward(self) -> bool:
        """Check whether this operation is ready to forward."""
        # - check input nodes' status
        status = [node.status == "finished" for node in self.input_nodes]
        if all(status):
            return True
        else:
            return False

    @abc.abstractmethod
    def forward(self):
        """"""
        # - set working directory and logger
        if not self.directory.exists():
            self.directory.mkdir(parents=True)

        return


if __name__ == "__main__":
    ...
