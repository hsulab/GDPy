#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import abc
import copy
import logging
import pathlib

from gdpx.core.component import BaseComponent
from gdpx.execution.service import ExecutionWorker, WorkerExecutionService
from gdpx.structures.builders.builder import StructureBuilder


class BaseExploration(BaseComponent):

    #: Name of the exploration.
    name: str = "exploration"

    _pending_runtime = None

    def register_runtime(self, runtime) -> None:
        """Keep a serializable runtime without loading execution providers.

        Scheduled explorations only need this configuration on the submission
        host. Worker construction and algorithm-specific runtime validation
        happen in the process that executes the exploration.
        """
        from gdpx.providers.configuration import RuntimeConfig

        config = runtime if isinstance(runtime, RuntimeConfig) else RuntimeConfig.from_mapping(runtime)
        parameters = config.to_dict()
        potential = parameters["potential"].get("parameters", {})
        # Preserve model aliases, but anchor existing local checkpoints before
        # the scheduler changes into the exploration output directory.
        def model_path(value):
            if isinstance(value, (str, pathlib.Path)) and pathlib.Path(value).is_file():
                return str(pathlib.Path(value).resolve())
            return value

        if "model" in potential:
            model = potential["model"]
            potential["model"] = ([model_path(item) for item in model]
                                  if isinstance(model, list) else model_path(model))
        self._pending_runtime = parameters

    def materialize_worker(self) -> None:
        """Bind a deferred runtime once execution starts."""
        if self._pending_runtime is not None:
            from gdpx.execution.factory import create_worker

            self.register_worker(create_worker(self._pending_runtime))
            self._pending_runtime = None

    def runtime_as_dict(self):
        """Serialize a deferred or already constructed calculation worker."""
        if self._pending_runtime is not None:
            return copy.deepcopy(self._pending_runtime)
        return self.worker.as_dict()

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
