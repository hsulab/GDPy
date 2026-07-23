#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import abc
import logging
from typing import Union

from gdpx.core.component import BaseComponent
from gdpx.builder.builder import StructureBuilder
from gdpx.worker.drive import DriverBasedWorker
from gdpx.worker.single import SingleWorker


class BaseExpedition(BaseComponent):

    #: Name of the expedition.
    name: str = "expedition"

    @abc.abstractmethod
    def read_convergence(self) -> bool: ...

    @abc.abstractmethod
    def get_workers(self) -> list[Union[DriverBasedWorker, SingleWorker]]: ...

    def run(self, *args, **kwargs) -> None:
        """"""
        # - some imported packages change `logging.basicConfig`
        #   and accidently add a StreamHandler to logging.root
        #   so remove it...
        for h in logging.root.handlers:
            if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler):
                logging.root.removeHandler(h)

        assert self.worker is not None, f"{self.name} has not set its worker properly."

        return

    def register_builder(self, builder: StructureBuilder) -> None:
        """Attach an already constructed builder."""
        if not isinstance(builder, StructureBuilder):
            raise TypeError(f"Expected StructureBuilder, got {type(builder).__name__}.")
        self.builder = builder

        return

    def register_worker(self, worker, *args, **kwargs) -> None:
        """Attach an already constructed worker."""
        if isinstance(worker, list):
            if not worker:
                raise ValueError("Cannot register an empty worker list.")
            worker = worker[0]
        if not isinstance(worker, (DriverBasedWorker, SingleWorker)):
            raise TypeError(f"Expected a worker instance, got {type(worker).__name__}.")
        self.worker = worker

        return


if __name__ == "__main__":
    ...
