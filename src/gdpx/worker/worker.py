#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import abc
import copy
import pathlib
from typing import NoReturn, Optional, Union, Callable, List, Tuple

import numpy as np

from tinydb import TinyDB, Query

from .. import config
from ..scheduler.scheduler import AbstractScheduler


"""worker = driver + scheduler.

A worker that manages a series of dynamics tasks
worker needs a scheduler to dertermine whether run by serial
or on cluster.

"""


class AbstractWorker(abc.ABC):
    """"""

    UUIDLEN = 36  # length of uuid

    _print: Callable = config._print
    _debug: Callable = config._debug

    # Number of executation in each queue job.
    batchsize: int = 1

    _directory: Optional[pathlib.Path] = None
    _scheduler = None
    _database = None

    _submit = True

    _script_name = "run.script"

    def __init__(
        self,
        directory: Optional[Union[str, pathlib.Path]] = None,
        batchsize=1,
        *args,
        **kwargs,
    ) -> None:
        """ """
        # - set default directory
        if directory is not None:
            self.directory = directory

        self.batchsize = batchsize

        self.n_jobs = config.NJOBS

        return

    @property
    def directory(self) -> pathlib.Path:
        """"""

        return self._directory

    @directory.setter
    def directory(self, directory_: Union[str, pathlib.Path]):
        self._directory = pathlib.Path(directory_)

        return

    @property
    def scheduler(self):

        return self._scheduler

    @scheduler.setter
    def scheduler(self, scheduler_):
        """"""
        assert isinstance(scheduler_, AbstractScheduler), ""
        self._scheduler = scheduler_

        # - update mode
        # if self._scheduler.name == "local"

        return

    def _split_groups(self, npoints: int) -> Tuple[List[int], List[int]]:
        """Split nframes into groups."""
        # - split frames
        ngroups = int(np.floor(1.0 * npoints / self.batchsize))
        group_indices = [0]
        for i in range(ngroups):
            group_indices.append((i + 1) * self.batchsize)
        if group_indices[-1] != npoints:
            group_indices.append(npoints)
        starts, ends = group_indices[:-1], group_indices[1:]
        assert len(starts) == len(ends), "Inconsistent start and end indices..."
        # group_indices = [f"{s}:{e}" for s, e in zip(starts,ends)]

        return (starts, ends)

    # @property
    # def database(self):

    #    return self._database

    # @database.setter
    # def database(self, database_):
    #    self._database = database_

    #    return

    def _initialise(self, *args, **kwargs):
        """"""
        if not self.directory.exists():
            self.directory.mkdir(parents=True, exist_ok=True)  # NOTE: ./tmp_folder
        else:
            ...
        assert self.directory, "Working directory is not set properly..."

        return

    @abc.abstractmethod
    def run(self, *args, **kwargs):
        """"""
        self._initialise(*args, **kwargs)
        # - some imported packages change `logging.basicConfig`
        #   and accidently add a StreamHandler to logging.root
        #   so remove it...
        # for h in logging.root.handlers:
        #    if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler):
        #        logging.root.removeHandler(h)
        self._print(f"~~~{self.__class__.__name__}+run")
        return

    def inspect(self, resubmit=False, *args, **kwargs):
        """"""
        self._initialise(*args, **kwargs)
        self._debug(f"~~~{self.__class__.__name__}+inspect")

        running_jobs = self._get_running_jobs()

        return

    def retrieve(self, *args, **kwargs):
        """"""
        self.inspect(*args, **kwargs)
        self._print(f"~~~{self.__class__.__name__}+retrieve")

        results = []

        return results

    def _get_running_jobs(self):
        """"""
        with TinyDB(
            self.directory / f"_{self.scheduler.name}_jobs.json", indent=2
        ) as database:
            running_jobs = database.search(
                Query().queued.exists() & (~Query().finished.exists())
            )
        running_jobs = [r["gdir"] for r in running_jobs]

        return running_jobs

    def _get_finished_jobs(self):
        """"""
        with TinyDB(
            self.directory / f"_{self.scheduler.name}_jobs.json", indent=2
        ) as database:
            finished_jobs = database.search(
                Query().queued.exists() & (Query().finished.exists())
            )
        finished_jobs = [r["gdir"] for r in finished_jobs]

        return finished_jobs

    def _get_retrieved_jobs(self):
        """"""
        with TinyDB(
            self.directory / f"_{self.scheduler.name}_jobs.json", indent=2
        ) as database:
            retrieved_jobs = database.search(
                Query().queued.exists()
                & (Query().finished.exists())
                & Query().retrieved.exists()
            )
        retrieved_jobs = [r["gdir"] for r in retrieved_jobs]

        return retrieved_jobs

    def _get_unretrieved_jobs(self):
        """"""
        with TinyDB(
            self.directory / f"_{self.scheduler.name}_jobs.json", indent=2
        ) as database:
            unretrieved_jobs = database.search(
                (Query().queued.exists() & Query().finished.exists())
                & (~Query().retrieved.exists())
            )
        unretrieved_jobs = [r["gdir"] for r in unretrieved_jobs]

        return unretrieved_jobs

    def get_number_of_running_jobs(self):
        """"""
        running_jobs = self._get_running_jobs()

        return len(running_jobs)

    def as_dict(self) -> dict:
        """"""
        worker_params = {}
        worker_params["potter"] = self.potter.as_dict()
        worker_params["driver"] = self.driver.as_dict()
        worker_params["scheduler"] = self.scheduler.as_dict()

        worker_params = copy.deepcopy(worker_params)

        return worker_params


if __name__ == "__main__":
    ...
