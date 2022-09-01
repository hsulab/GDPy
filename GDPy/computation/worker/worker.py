#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" worker = driver + scheduler
    A worker that manages a series of dynamics tasks
    worker needs a scheduler to dertermine whether run by serial
    or on cluster
"""

import abc
import copy
import pathlib
import logging

from tinydb import TinyDB, Query

from GDPy.scheduler.factory import create_scheduler
from GDPy.potential.manager import PotManager
from GDPy.scheduler.scheduler import AbstractScheduler


class AbstractWorker(abc.ABC):

    """ - components
        single-frame methods
            Monte Carlo
        population methods
            GA = (generator + propagator) + driver + scheduler
        - machines
            moniter
            job
    """

    restart = True
    logger = None

    _directory = None
    _scheduler = None
    _database = None

    _submit = True

    prefix = "worker"
    worker_status = dict(queued=[], finished=[])

    def __init__(self, params, directory_=None) -> None:
        """
        """
        # - pop some
        self.prefix = params.pop("prefix", "worker")

        # - create scheduler
        scheduler_params = params.pop("scheduler", {})
        self.scheduler = create_scheduler(scheduler_params)

        # - potter and driver
        params_ = copy.deepcopy(params)
        pot_dict = params_.get("potential", None)
        if pot_dict is None:
            raise RuntimeError("Need potential...")
        pm = PotManager() # main potential manager
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

        # NOTE: create a database
        self._database = TinyDB(self.directory/".metadata.json", indent=2)

        return
    
    @property
    def scheduler(self):

        return self._scheduler
    
    @scheduler.setter
    def scheduler(self, scheduler_):
        """"""
        assert isinstance(scheduler_, AbstractScheduler), ""
        self._scheduler = scheduler_

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
        self.database = TinyDB(self.directory/".metadata.json", indent=2)

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
        self.logger.info(f"@@@{self.__class__.__name__}")

        return
    
    @abc.abstractmethod
    def run(self, *args, **kwargs):
        """"""
        self._initialise(*args, **kwargs)
        return
    
    @abc.abstractmethod
    def retrieve(self, *args, **kwargs):
        """"""
        self._initialise(*args, **kwargs)
        return

    def _get_running_jobs(self):
        """"""
        running_jobs = self.database.search(
            Query().queued.exists() and (~Query().finished.exists())
        )
        running_jobs = [r["gdir"] for r in running_jobs]

        return running_jobs
    
    def get_number_of_running_jobs(self):
        """"""
        running_jobs = self._get_running_jobs()

        return len(running_jobs)


if __name__ == "__main__":
    pass