#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import abc
import pathlib

from gdpx import config

class AbstractExpedition(abc.ABC):

    _print = config._print
    _debug = config._debug

    @property
    def directory(self):

        return self._directory

    @directory.setter
    def directory(self, directory_):
        self._directory = pathlib.Path(directory_)

        return

    @abc.abstractmethod
    def read_convergence(self):

        return

    @abc.abstractmethod
    def get_workers(self):

        return


if __name__ == "__main__":
    ...