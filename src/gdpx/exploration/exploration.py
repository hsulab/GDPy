#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import abc
import logging

from gdpx.core.component import BaseComponent
from gdpx.execution.service import ExecutionWorker, WorkerExecutionService
from gdpx.structures.builders.builder import StructureBuilder


class BaseExploration(BaseComponent):

    #: Name of the exploration.
    name: str = "exploration"

    @abc.abstractmethod
    def read_convergence(self) -> bool: ...

    @abc.abstractmethod
    def get_workers(self) -> list[ExecutionWorker]: ...

    def restore_continuation(self, continuation) -> None:
        """Restore provider-owned state from the preceding workflow iteration."""
        if continuation is not None:
            raise TypeError(f"{type(self).__name__} does not support workflow continuation.")

    def capture_continuation(self):
        """Return provider-owned state for the next workflow iteration."""
        return None

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
        if not isinstance(worker, ExecutionWorker):
            raise TypeError(f"Expected a worker instance, got {type(worker).__name__}.")
        self.worker = worker
        self.execution = WorkerExecutionService(worker)

        return


if __name__ == "__main__":
    ...
