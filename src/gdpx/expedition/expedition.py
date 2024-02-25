#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import abc
import copy
import pathlib

from . import config
from . import registers
from . import ComputerVariable, DriverBasedWorker

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

    def register_worker(self, worker: dict, *args, **kwargs):
        """"""
        if isinstance(worker, dict):
            worker_params = copy.deepcopy(worker)
            worker = registers.create(
                "variable", "computer", convert_name=True, **worker_params
            ).value[0]
        elif isinstance(worker, list): # assume it is from a computervariable
            worker = worker[0]
        elif isinstance(worker, ComputerVariable):
            worker = worker.value[0]
        elif isinstance(worker, DriverBasedWorker):
            worker = worker
        else:
            raise RuntimeError(f"Unknown worker type {worker}")
        
        self.worker = worker

        return


if __name__ == "__main__":
    ...