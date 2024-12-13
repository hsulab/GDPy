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


def parse_worker(inp_worker: dict, *args, **kwargs):
    """Parse DriverBasedWorker for this expedition."""
    worker = None
    if isinstance(inp_worker, dict):
        worker_params = copy.deepcopy(inp_worker)
        worker = registers.create(
            "variable", "computer", convert_name=True, **worker_params
        ).value[0]
    elif isinstance(inp_worker, list):  # assume it is from a computervariable
        worker = inp_worker[0]
    elif isinstance(inp_worker, ComputerVariable):
        worker = inp_worker.value[0]
    elif isinstance(inp_worker, DriverBasedWorker) or isinstance(inp_worker, SingleWorker):
        worker = inp_worker
    else:
        raise RuntimeError(f"Unknown worker type {inp_worker}.")

    return worker


def canonicalise_builder(builder: dict):
    """"""
    if isinstance(builder, dict):
        builder_params = copy.deepcopy(builder)
        builder_method = builder_params.pop("method")
        builder = registers.create(
            "builder", builder_method, convert_name=False, **builder_params
        )
    else:  # Assume it is already a StructureBuilder
        builder = builder

    return builder


class AbstractExpedition(AbstractNode):

    #: Name of the expedition.
    name: str = "expedition"

    @abc.abstractmethod
    def read_convergence(self) -> bool:

        ...

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

        assert self.worker is not None, f"{self.name} has not set its worker properly."

        return

    def register_builder(self, builder: dict) -> None:
        """Register StructureBuilder for this expedition."""
        if isinstance(builder, dict):
            builder_params = copy.deepcopy(builder)
            builder_method = builder_params.pop("method")
            builder = registers.create(
                "builder", builder_method, convert_name=False, **builder_params
            )
        else:
            builder = builder

        self.builder = builder

        return

    def register_worker(self, worker: dict, *args, **kwargs) -> None:
        """Register DriverBasedWorker for this expedition."""
        self.worker = parse_worker(inp_worker=worker)

        return


if __name__ == "__main__":
    ...
