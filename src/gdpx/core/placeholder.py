#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pathlib
from typing import NoReturn, Union

class Placeholder: # Placeholder

    """Placeholder for input structures that may be from external files.
    """

    #: Working directory for the operation.
    _directory: Union[str,pathlib.Path] = pathlib.Path.cwd()

    #: Working status that should be always finished.
    status = "finished"

    def __init__(self):
        """"""
        self.consumers = []

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

    def reset(self):
        """Reset node's output and status."""
        ...

        return


if __name__ == "__main__":
    ...
