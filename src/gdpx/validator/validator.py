#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import abc
import logging
import pathlib
from typing import NoReturn, Union, Callable

from gdpx import config


class AbstractValidator(abc.ABC):

    _print: Callable = config._print
    _debug: Callable = config._debug

    _directory = pathlib.Path.cwd()

    def __init__(self, directory: Union[str,pathlib.Path]="./", *args, **kwargs):
        """
        """
        self.directory = directory

        self.njobs = config.NJOBS

        return
    
    @property
    def directory(self):
        """"""

        return self._directory

    @directory.setter
    def directory(self, directory_):
        """"""
        directory_ = pathlib.Path(directory_)
        self._directory = directory_

        return

    @abc.abstractmethod
    def run(self, *args, **kwargs):
        """"""
        if not self.directory.exists():
            self.directory.mkdir(parents=True)

        return


if __name__ == "__main__":
    ...