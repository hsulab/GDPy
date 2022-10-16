#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import abc
import uuid
import copy
import pathlib
import logging

from tinydb import TinyDB, Query

from GDPy.scheduler import create_scheduler
from GDPy.potential.register import PotentialRegister
from GDPy.scheduler.scheduler import AbstractScheduler


"""worker = driver + scheduler.

A worker that manages a series of dynamics tasks
worker needs a scheduler to dertermine whether run by serial
or on cluster.

"""


class AbstractWorker(abc.ABC):

    """"""

    UUIDLEN = 36 # length of uuid

    restart = True
    logger = None

    _directory = None
    _scheduler = None
    _database = None

    _submit = True

    _exec_mode = "queue"

    def __init__(self, params, directory_=None) -> None:
        """
        """
        # - create scheduler
        scheduler_params = params.pop("scheduler", {})
        self.scheduler = create_scheduler(scheduler_params)

        # - potter and driver
        params_ = copy.deepcopy(params)
        pot_dict = params_.get("potential", None)
        if pot_dict is None:
            raise RuntimeError("Need potential...")
        pm = PotentialRegister() # main potential manager
        potter = pm.create_potential(pot_name = pot_dict["name"])
        potter.register_calculator(pot_dict["params"])
        potter.version = pot_dict.get("version", "unknown") # NOTE: important for calculation in exp

        self.driver = potter.create_driver(params_["driver"])

        # - set default directory
        #self.directory = self.directory / "MyWorker" # TODO: set dir
        if directory_:
            self.directory = directory_
        
        return

    @property
    def directory(self):

        return self._directory
    
    @directory.setter
    def directory(self, directory_):
        """"""
        # - create main dir
        directory_ = pathlib.Path(directory_)
        if not directory_.exists():
            directory_.mkdir() # NOTE: ./tmp_folder
        else:
            pass
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
    
    @property
    def database(self):

        return self._database
    
    @database.setter
    def database(self, database_):
        self._database = database_

        return 
    
    def _init_database(self):
        """"""
        self.database = TinyDB(
            self.directory/f"_{self.scheduler.name}_jobs.json", indent=2
        )

        return
    
    def _init_logger(self):
        """"""
        self.logger = logging.getLogger(__name__)

        log_level = logging.INFO

        self.logger.setLevel(log_level)

        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        working_directory = self.directory
        log_fpath = working_directory / (self.__class__.__name__+".out")

        if self.restart:
            fh = logging.FileHandler(filename=log_fpath, mode="a")
        else:
            fh = logging.FileHandler(filename=log_fpath, mode="w")

        fh.setLevel(log_level)
        #fh.setFormatter(formatter)

        ch = logging.StreamHandler()
        ch.setLevel(log_level)
        #ch.setFormatter(formatter)

        self.logger.addHandler(ch)
        self.logger.addHandler(fh)

        return
    
    def _initialise(self, *args, **kwargs):
        """"""
        assert self.directory, "Working directory is not set properly..."
        self._init_database()
        if self.logger is None:
            self._init_logger()

        return
    
    @abc.abstractmethod
    def run(self, *args, **kwargs):
        """"""
        self._initialise(*args, **kwargs)
        self.logger.info(f"@@@{self.__class__.__name__}+run")
        return

    def inspect(self, *args, **kwargs):
        """ check if any job were finished
        """
        self._initialise(*args, **kwargs)
        self.logger.info(f"@@@{self.__class__.__name__}+inspect")

        scheduler = self.scheduler

        running_jobs = self._get_running_jobs()
        for job_name in running_jobs:
            group_directory = self.directory / job_name[self.UUIDLEN+1:]
            scheduler.set(**{"job-name": job_name})
            scheduler.script = group_directory/"run-driver.script" 

            info_name = job_name[self.UUIDLEN+1:]
            if scheduler.is_finished():
                self.logger.info(f"{info_name} is finished...")
                doc_data = self.database.get(Query().gdir == job_name)
                self.database.update({"finished": True}, doc_ids=[doc_data.doc_id])
            else:
                self.logger.info(f"{info_name} is running...")

        return
    
    def retrieve(self, *args, **kwargs):
        """"""
        self.inspect(*args, **kwargs)
        self.logger.info(f"@@@{self.__class__.__name__}+retrieve")

        gdirs, results = [], []

        # - check status and get latest results
        unretrieved_jobs = self._get_unretrieved_jobs()
        for job_name in unretrieved_jobs:
            # NOTE: sometimes prefix has number so confid may be striped
            group_directory = self.directory / job_name[self.UUIDLEN+1:]
            gdirs.append(group_directory)

        if gdirs:
            results = self._read_results(gdirs, *args, **kwargs)

        for job_name in unretrieved_jobs:
            doc_data = self.database.get(Query().gdir == job_name)
            self.database.update({"retrieved": True}, doc_ids=[doc_data.doc_id])

        return results
    
    @abc.abstractmethod
    def _read_results(self, gdirs, *args, **kwargs):
        """"""
        return

    def _get_running_jobs(self):
        """"""
        self._initialise()
        running_jobs = self.database.search(
            Query().queued.exists() & (~Query().finished.exists())
        )
        running_jobs = [r["gdir"] for r in running_jobs]

        return running_jobs

    def _get_finished_jobs(self):
        """"""
        self._initialise()
        finished_jobs = self.database.search(
            Query().queued.exists() & (Query().finished.exists())
        )
        finished_jobs = [r["gdir"] for r in finished_jobs]

        return finished_jobs
    
    def _get_retrieved_jobs(self):
        """"""
        self._initialise()
        retrieved_jobs = self.database.search(
            Query().queued.exists() & (Query().finished.exists()) &
            Query().retrieved.exists()
        )
        retrieved_jobs = [r["gdir"] for r in retrieved_jobs]

        return retrieved_jobs
    
    def _get_unretrieved_jobs(self):
        """"""
        self._initialise()
        unretrieved_jobs = self.database.search(
            (Query().queued.exists() & Query().finished.exists()) &
            (~Query().retrieved.exists())
        )
        unretrieved_jobs = [r["gdir"] for r in unretrieved_jobs]

        return unretrieved_jobs
    
    def get_number_of_running_jobs(self):
        """"""
        self._initialise()
        running_jobs = self._get_running_jobs()

        return len(running_jobs)


if __name__ == "__main__":
    pass
