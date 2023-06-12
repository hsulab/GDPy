#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pathlib
import uuid
import yaml

from tinydb import Query

from .worker import AbstractWorker
from ..scheduler.scheduler import AbstractScheduler

"""Worker that manages routines.

Since a routine is made up of several basic workers, this worker is a monitor that 
tracks its progress.

"""

class RoutineBasedWorker(AbstractWorker):

    batchsize: int = 1

    _script_name: str = "run.script"

    def __init__(self, routine, scheduler: AbstractScheduler, batchsize: int=1, directory=None) -> None:
        """"""
        super().__init__(directory)

        self.routine = routine
        self.scheduler = scheduler

        self.batchsize = batchsize
        self.wait_time = 60

        return

    def run(self, builder=None, *args, **kwargs) -> None:
        """"""
        super().run(*args, **kwargs)

        routine = self.routine
        scheduler = self.scheduler

        if self.scheduler.name == "local":
            routine.directory = self.directory
            routine.run()
        else:
            size = 1
            for i in range(size):
                uid = str(uuid.uuid1())
                job_name = uid + "-" + "routine" + "-" + f"{i}"
                wdir = self.directory / ("routine" + "-" + f"{i}")
                if not wdir.exists():
                    wdir.mkdir(parents=True)
                else:
                    # TODO: better check
                    continue

                routine_params = routine.as_dict()
                with open(wdir/(f"routine-{uid}.yaml"), "w") as fopen:
                    yaml.safe_dump(routine_params, fopen)
            
                scheduler.job_name = job_name
                scheduler.script = wdir/self._script_name
                scheduler.user_commands = "gdp routine {} --wait {}".format(
                    str(wdir/f"routine-{uid}.yaml"), self.wait_time
                )
                scheduler.write()
                if self._submit:
                    self.logger.info(f"{wdir.name}: {scheduler.submit()}")
                else:
                    self.logger.info(f"{wdir.name} waits to submit.")

                # - update database
                _ = self.database.insert(
                    dict(
                        uid = uid,
                        gdir=job_name, 
                        group_number=i, 
                        wdir_names=[wdir.name], 
                        queued=True
                    )
                )

        return
    
    def inspect(self, resubmit=False, *args, **kwargs):
        """"""
        self._initialise(*args, **kwargs)
        self._debug(f"@@@{self.__class__.__name__}+inspect")

        running_jobs = self._get_running_jobs()
        for job_name in running_jobs:
            doc_data = self.database.get(Query().gdir == job_name)
            uid = doc_data["uid"]

            print("doc_data: ", doc_data)
            wdir = self.directory/doc_data["wdir_names"][0]

            self.scheduler.job_name = job_name
            self.scheduler.script = wdir/"train.script"
            
            if self.scheduler.is_finished():
                # -- check if the job finished properly
                self.routine.directory = wdir
                if self.routine.read_convergence():
                    self.database.update({"finished": True}, doc_ids=[doc_data.doc_id])
                    self.logger.info(f"{job_name} finished.")
                else:
                    if resubmit:
                        jobid = self.scheduler.submit()
                        self.logger.info(f"{job_name} is re-submitted with JOBID {jobid}.")
            else:
                self._print(f"{job_name} is running...")

        return
    
    def retrieve(self, ignore_retrieved: bool=False, *args, **kwargs):
        """"""
        #raise NotImplementedError(f"{self.__class__.__name__}")
        self.inspect(*args, **kwargs)
        self._debug(f"@@@{self.__class__.__name__}+retrieve")

        unretrieved_wdirs_ = []
        if not ignore_retrieved:
            unretrieved_jobs = self._get_unretrieved_jobs()
        else:
            unretrieved_jobs = self._get_finished_jobs()

        for job_name in unretrieved_jobs:
            doc_data = self.database.get(Query().gdir == job_name)
            unretrieved_wdirs_.extend(
                (self.directory/w).resolve() for w in doc_data["wdir_names"]
            )
        unretrieved_wdirs = unretrieved_wdirs_

        workers = []
        if unretrieved_wdirs:
            unretrieved_wdirs = [pathlib.Path(x) for x in unretrieved_wdirs]
            print("unretrieved_wdirs: ", unretrieved_wdirs)
            for p in unretrieved_wdirs:
                self.routine.directory = p
                workers.extend(self.routine.get_workers())

        for job_name in unretrieved_jobs:
            doc_data = self.database.get(Query().gdir == job_name)
            self.database.update({"retrieved": True}, doc_ids=[doc_data.doc_id])

        return workers

if __name__ == "__main__":
    ...