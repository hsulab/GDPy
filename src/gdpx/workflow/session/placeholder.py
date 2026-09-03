#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import pathlib

from .registry import PLACEHOLDER_REGISTRY
from .node import NodeKind


class Placeholder:
    """Placeholder for input structures that may be from external files."""

    #: Working status that should be always finished.
    status = "finished"
    node_kind = NodeKind.PLACEHOLDER

    def __init__(self):
        """"""
        #: The input nodes.
        self.consumers = []

        #: Working directory for the operation.
        self._directory: pathlib.Path = pathlib.Path.cwd()

        return

    @property
    def directory(self) -> pathlib.Path:
        """"""

        return self._directory

    @directory.setter
    def directory(self, directory) -> None:
        """"""
        self._directory = pathlib.Path(directory)

        return

    def reset(self):
        """Reset node's output and status."""
        ...

        return


if __name__ == "__main__":
    ...
