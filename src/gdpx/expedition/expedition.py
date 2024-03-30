#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import abc
import copy
import logging
import pathlib

from . import config
from . import registers
from . import ComputerVariable, DriverBasedWorker, SingleWorker

from ..core.node import AbstractNode


class AbstractExpedition(AbstractNode):

    #: Name of the expedition.
    name: str = "expedition"

    @abc.abstractmethod
    def read_convergence(self):

        return

    @abc.abstractmethod
    def get_workers(self):

        return

    def run(self, *args, **kwargs) -> None:
        """"""
        # - some imported packages change `logging.basicConfig`
        #   and accidently add a StreamHandler to logging.root
        #   so remove it...
        for h in logging.root.handlers:
            if isinstance(h, logging.StreamHandler) and not isinstance(
                h, logging.FileHandler
            ):
                logging.root.removeHandler(h)

        assert self.worker is not None, "MC has not set its worker properly."

        return

    def register_worker(self, worker: dict, *args, **kwargs):
        """"""
        if isinstance(worker, dict):
            worker_params = copy.deepcopy(worker)
            worker = registers.create(
                "variable", "computer", convert_name=True, **worker_params
            ).value[0]
        elif isinstance(worker, list):  # assume it is from a computervariable
            worker = worker[0]
        elif isinstance(worker, ComputerVariable):
            worker = worker.value[0]
        elif isinstance(worker, DriverBasedWorker) or isinstance(worker, SingleWorker):
            worker = worker
        else:
            raise RuntimeError(f"Unknown worker type {worker}")

        self.worker = worker

        return


if __name__ == "__main__":
    ...
