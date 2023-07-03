#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import abc
import uuid
import copy
import pathlib
from typing import NoReturn, Callable, List, Tuple
import logging

import numpy as np

from tinydb import TinyDB, Query

from GDPy import config
from GDPy.scheduler.scheduler import AbstractScheduler


"""worker = driver + scheduler.

A worker that manages a series of dynamics tasks
worker needs a scheduler to dertermine whether run by serial
or on cluster.

"""


class AbstractWorker(abc.ABC):

    """"""

    UUIDLEN = 36 # length of uuid

    _print: Callable = config._print
    _debug: Callable = config._debug

    # Number of executation in each queue job.
    batchsize: int = 1

    _directory = None
    _scheduler = None
    _database = None

    _submit = True

    _script_name = "run.script"

    def __init__(self, directory=None, batchsize=1, *args, **kwargs) -> None:
        """
        """
        # - set default directory
        if directory is not None:
            self.directory = directory

        self.batchsize = batchsize

        self.n_jobs = config.NJOBS
        
        return

    @property
    def directory(self):

        return self._directory
    
    @directory.setter
    def directory(self, directory_):
        """"""
        # - create main dir
        directory_ = pathlib.Path(directory_)
        self._directory = directory_

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
        #if self._scheduler.name == "local"

        return

    def _split_groups(self, npoints: int) -> Tuple[List[int],List[int]]:
        """Split nframes into groups."""
        # - split frames
        ngroups = int(np.floor(1.*npoints/self.batchsize))
        group_indices = [0]
        for i in range(ngroups):
            group_indices.append((i+1)*self.batchsize)
        if group_indices[-1] != npoints:
            group_indices.append(npoints)
        starts, ends = group_indices[:-1], group_indices[1:]
        assert len(starts) == len(ends), "Inconsistent start and end indices..."
        #group_indices = [f"{s}:{e}" for s, e in zip(starts,ends)]

        return (starts, ends)
    
    #@property
    #def database(self):

    #    return self._database
    
    #@database.setter
    #def database(self, database_):
    #    self._database = database_

    #    return 

    
    def _initialise(self, *args, **kwargs):
        """"""
        if not self.directory.exists():
            self.directory.mkdir(parents=True, exist_ok=True) # NOTE: ./tmp_folder
        else:
            ...
        assert self.directory, "Working directory is not set properly..."

        return
    
    @abc.abstractmethod
    def run(self, *args, **kwargs):
        """"""
        self._initialise(*args, **kwargs)
        self._print(f"~~~{self.__class__.__name__}+run")
        return

    def inspect(self, *args, **kwargs):
        """ check if any job were finished
        """
        self._initialise(*args, **kwargs)
        self._print(f"~~~{self.__class__.__name__}+inspect")

        scheduler = self.scheduler

        running_jobs = self._get_running_jobs()
        for job_name in running_jobs:
            group_directory = self.directory / job_name[self.UUIDLEN+1:]
            scheduler.set(**{"job-name": job_name})
            scheduler.script = group_directory/"run-driver.script" 

            info_name = job_name[self.UUIDLEN+1:]
            if scheduler.is_finished():
                self._print(f"{info_name} at {self.directory.name} is finished...")
                with TinyDB(
                    self.directory/f"_{self.scheduler.name}_jobs.json", indent=2
                ) as database:
                    doc_data = database.get(Query().gdir == job_name)
                    database.update({"finished": True}, doc_ids=[doc_data.doc_id])
            else:
                self._print(f"{info_name} at {self.directory.name} is running...")

        return

    def inspect(self, resubmit=False, *args, **kwargs):
        """"""
        self._initialise(*args, **kwargs)
        self._debug(f"~~~{self.__class__.__name__}+inspect")

        running_jobs = self._get_running_jobs()

        with TinyDB(
            self.directory/f"_{self.scheduler.name}_jobs.json", indent=2
        ) as database:
            for job_name in running_jobs:
                doc_data = database.get(Query().gdir == job_name)
                uid = doc_data["uid"]

                self.scheduler.job_name = job_name
                self.scheduler.script = self.directory/self._script_name

                if self.scheduler.is_finished():
                    # -- check if the job finished properly
                    # read_convergence
                    if True:
                        database.update({"finished": True}, doc_ids=[doc_data.doc_id])
                    else:
                        if resubmit:
                            jobid = self.scheduler.submit()
                            self._print(f"{job_name} is re-submitted with JOBID {jobid}.")
                else:
                    self._print(f"{job_name} is running...")

        return
    
    def retrieve(self, *args, **kwargs):
        """"""
        self.inspect(*args, **kwargs)
        self._print(f"~~~{self.__class__.__name__}+retrieve")

        gdirs, results = [], []

        # - check status and get latest results
        unretrieved_jobs = self._get_unretrieved_jobs()
        for job_name in unretrieved_jobs:
            # NOTE: sometimes prefix has number so confid may be striped
            group_directory = self.directory / job_name[self.UUIDLEN+1:]
            gdirs.append(group_directory)

        if gdirs:
            results = self._read_results(gdirs, *args, **kwargs)

        with TinyDB(
            self.directory/f"_{self.scheduler.name}_jobs.json", indent=2
        ) as database:
            for job_name in unretrieved_jobs:
                doc_data = database.get(Query().gdir == job_name)
                database.update({"retrieved": True}, doc_ids=[doc_data.doc_id])

        return results

    def _get_running_jobs(self):
        """"""
        with TinyDB(
            self.directory/f"_{self.scheduler.name}_jobs.json", indent=2
        ) as database:
            running_jobs = database.search(
                Query().queued.exists() & (~Query().finished.exists())
            )
        running_jobs = [r["gdir"] for r in running_jobs]

        return running_jobs

    def _get_finished_jobs(self):
        """"""
        with TinyDB(
            self.directory/f"_{self.scheduler.name}_jobs.json", indent=2
        ) as database:
            finished_jobs = database.search(
                Query().queued.exists() & (Query().finished.exists())
            )
        finished_jobs = [r["gdir"] for r in finished_jobs]

        return finished_jobs
    
    def _get_retrieved_jobs(self):
        """"""
        with TinyDB(
            self.directory/f"_{self.scheduler.name}_jobs.json", indent=2
        ) as database:
            retrieved_jobs = database.search(
                Query().queued.exists() & (Query().finished.exists()) &
                Query().retrieved.exists()
            )
        retrieved_jobs = [r["gdir"] for r in retrieved_jobs]

        return retrieved_jobs
    
    def _get_unretrieved_jobs(self):
        """"""
        with TinyDB(
            self.directory/f"_{self.scheduler.name}_jobs.json", indent=2
        ) as database:
            unretrieved_jobs = database.search(
                (Query().queued.exists() & Query().finished.exists()) &
                (~Query().retrieved.exists())
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
