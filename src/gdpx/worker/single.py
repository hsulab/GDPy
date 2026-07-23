#!/usr/bin/env python3
# -*- coding: utf-8 -*

"""Single-structure computation worker.

This is a thin wrapper around ``DriverBasedWorker`` that constrains
the worker to process exactly one structure.  It preserves the
original :class:`SingleWorker` API for backward compatibility.
"""


import pathlib
from typing import Optional, Union

from tinydb import Query, TinyDB

from gdpx.computation.driver import BaseDriver
from gdpx.worker.registry import WORKER_REGISTRY
from gdpx.potential.manager import BasePotentialManager
from gdpx.scheduler.scheduler import BaseScheduler

from .drive import DriverBasedWorker
from .pairing import Pairing


@WORKER_REGISTRY.register
class SingleWorker(DriverBasedWorker):
    """Worker that accepts only a single structure.

    This is a convenience subclass of :class:`DriverBasedWorker` with
    ``batchsize=1`` and a few extra methods for MC-style workflows
    (e.g. :meth:`rewind_to_step`).
    """

    COMP_PREFIX: str = "cand"

    _retrieve_mode: str = "single"

    def __init__(
        self,
        potter: Optional[BasePotentialManager] = None,
        driver: Optional[Union[BaseDriver, list[BaseDriver]]] = None,
        scheduler: Optional[BaseScheduler] = None,
        directory: Union[str, pathlib.Path] = "./",
    ) -> None:
        super().__init__(
            potter=potter,
            driver=driver,
            scheduler_=scheduler,
            directory=directory,
            batchsize=1,
            pairing=Pairing.BROADCAST,
        )
        self._wdir_name: str = ""

    @staticmethod
    def from_a_worker(worker: DriverBasedWorker) -> "SingleWorker":
        """Create a SingleWorker sharing its configuration with *worker*."""
        single = SingleWorker(
            potter=getattr(worker, "potter", None),
            driver=worker.drivers if worker.drivers else None,
            scheduler=worker.scheduler,
            directory=worker.directory,
        )
        single._share_wdir = worker._share_wdir
        return single

    @property
    def wdir_name(self) -> str:
        return self._wdir_name

    @wdir_name.setter
    def wdir_name(self, name: str):
        self._wdir_name = name

    def _prepare_batches(self, frames, start_confid, rng_states):
        if self._wdir_name:
            # Override auto-naming with the user-provided wdir name
            batches = super()._prepare_batches(frames, start_confid, rng_states)
            # Replace the first wdir name in each batch
            for batch in batches:
                if batch[1]:
                    batch[1][0] = self._wdir_name
            return batches
        return super()._prepare_batches(frames, start_confid, rng_states)

    def rewind_to_step(self, step: int):
        """Remove computation folders from the database after *step*.

        Used in MC expeditions when restarting from a checkpoint.
        """
        def test_func(wdir_names, step: int) -> bool:
            if not wdir_names:
                return False
            try:
                cand_index = int(pathlib.Path(wdir_names[0]).name[4:])
            except (ValueError, IndexError):
                return False
            return cand_index > step

        with TinyDB(self.directory / f"_{self.scheduler.name}_jobs.json", indent=2) as database:
            doc_data = database.search(Query().wdir_names.test(test_func, step))
            doc_ids = [doc.doc_id for doc in doc_data]
            if doc_ids:
                database.remove(doc_ids=doc_ids)

    def as_dict(self) -> dict:
        params = super().as_dict()
        params["use_single"] = True
        return params


if __name__ == "__main__":
    ...
