#!/usr/bin/env python3
# -*- coding: utf-8 -*

import pathlib
import time
import uuid
import yaml

from typing import List

from tinydb import Query, TinyDB

from ase import Atoms
from ase.io import read, write

from .worker import AbstractWorker
from ..computation.driver import AbstractDriver
from ..potential.manager import AbstractPotentialManager
from ..utils.command import CustomTimer
from ..scheduler.scheduler import AbstractScheduler
from ..scheduler.local import LocalScheduler


class GridDriverBasedWorker(AbstractWorker):

    def __init__(self, potter: List[AbstractPotentialManager], driver: AbstractDriver, scheduler: AbstractScheduler=LocalScheduler(), directory="./", *args, **kwargs) -> None:
        """"""
        super().__init__(directory)

        self.potter = potter # a List of potters
        self.driver = driver
        self.scheduler = scheduler

        return
    
    def run(self, builder, *args, **kwargs):
        """This worker accepts only a single structure."""
        super().run(*args, **kwargs)

        # - structures
        if isinstance(builder, list): # assume List[Atoms]
            frames = builder
        else: # assume it is a builder
            frames = builder.run()
        nframes = len(frames)
        assert len(frames) == 1, f"{self.__class__.__name__} accepts only a single structure."

        # - potentials
        driver_dict = self.driver.as_dict()
        drivers = [p.create_driver(driver_dict) for p in self.potter]
        ndrivers = len(drivers)

        # -

        uid = str(uuid.uuid1())
        wdir_names = [f"pott{i}" for i in range(ndrivers)]
        job_name = uid

        scheduler = self.scheduler
        if scheduler.name == "local":
            with CustomTimer(name="run-driver", func=self._print):
                for wdir_name, driver in zip(wdir_names, drivers):
                    driver.directory = self.directory / wdir_name
                    self._print(
                        f"{time.asctime( time.localtime(time.time()) )} {wdir_name} {driver.directory.name} is running..."
                    )
                    driver.reset()
                    driver.run(frames[0], read_ckpt=True, extra_info=None)
        else:
            worker_params = {}
            worker_params["use_single"] = True
            worker_params["driver"] = self.driver.as_dict()
            worker_params["potential"] = self.potter.as_dict()

            with open(self.directory/f"worker-{uid}.yaml", "w") as fopen:
                yaml.dump(worker_params, fopen)

            # - save structures
            dataset_path = str((self.directory/f"_gdp_inp.xyz").resolve())
            write(dataset_path, frames[0])

            # - save scheduler file
            jobscript_fname = f"run-{uid}.script"
            self.scheduler.job_name = job_name
            self.scheduler.script = self.directory/jobscript_fname

            self.scheduler.user_commands = "gdp -p {} compute {}\n".format(
                (self.directory/f"worker-{uid}.yaml").name, dataset_path
            )

            # - TODO: check whether params for scheduler is changed
            self.scheduler.write()
            if self._submit:
                self._print(f"{self.directory.name} JOBID: {self.scheduler.submit()}")
            else:
                self._print(f"{self.directory.name} waits to submit.")

        # - save this batch job to the database
        with TinyDB(
            self.directory/f"_{self.scheduler.name}_jobs.json", indent=2
        ) as database:
            _ = database.insert(
                dict(
                    uid = uid,
                    #md5 = identifier,
                    gdir=job_name, 
                    #group_number=ig, 
                    wdir_names=wdir_names,
                    queued=True
                )
            )

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
                self.scheduler.script = self.directory/"train.script"

                if self.scheduler.is_finished():
                    # -- check if the job finished properly
                    is_finished = False
                    wdir_names = doc_data["wdir_names"]
                    for x in wdir_names:
                        wdir = self.directory/x
                        if not wdir.exists():
                            break
                        else:
                            self.driver.directory = wdir
                            if not self.driver.read_convergence():
                                break
                    else:
                        is_finished = True
                    if is_finished:
                        self._print(f"{job_name} is finished.")
                        database.update({"finished": True}, doc_ids=[doc_data.doc_id])
                    else:
                        if resubmit:
                            jobid = self.scheduler.submit()
                            self._print(f"{job_name} is re-submitted with JOBID {jobid}.")
                else:
                    self._print(f"{job_name} is running...")

        return

    def retrieve(self, include_retrieved: bool=False, *args, **kwargs):
        """Retrieve training results.

        """
        self.inspect(*args, **kwargs)
        self._debug(f"~~~{self.__class__.__name__}+retrieve")

        unretrieved_wdirs_ = []
        if not include_retrieved:
            unretrieved_jobs = self._get_unretrieved_jobs()
        else:
            unretrieved_jobs = self._get_finished_jobs()

        with TinyDB(
            self.directory/f"_{self.scheduler.name}_jobs.json", indent=2
        ) as database:
            for job_name in unretrieved_jobs:
                doc_data = database.get(Query().gdir == job_name)
                unretrieved_wdirs_.extend(
                    (self.directory/w).resolve() for w in doc_data["wdir_names"]
                )
            unretrieved_wdirs = unretrieved_wdirs_

            results = []
            if unretrieved_wdirs:
                unretrieved_wdirs = [pathlib.Path(x) for x in unretrieved_wdirs]
                self._debug(f"unretrieved_wdirs: {unretrieved_wdirs}")
                for p in unretrieved_wdirs:
                    self.driver.directory = p
                    results.append(self.driver.read_trajectory())

            for job_name in unretrieved_jobs:
                doc_data = database.get(Query().gdir == job_name)
                database.update({"retrieved": True}, doc_ids=[doc_data.doc_id])

        return results
    
    def as_dict(self) -> dict:
        """"""
        params = super().as_dict()

        return params
    

if __name__ == "__main__":
    ...