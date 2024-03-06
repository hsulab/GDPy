#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pathlib
import uuid
import warnings
import yaml

from tinydb import Query, TinyDB

from .worker import AbstractWorker
from ..scheduler.scheduler import AbstractScheduler

"""Worker that manages expeditions.

Since an expedition is made up of several basic workers, this worker is a monitor that 
tracks its progress.

"""

class ExpeditionBasedWorker(AbstractWorker):

    #: Prefix of the expedition folder name.
    EXP_INDEX: str = "expedition"

    batchsize: int = 1

    _script_name: str = "run.script"

    def __init__(self, expedition, scheduler: AbstractScheduler, batchsize: int=1, directory=None) -> None:
        """"""
        super().__init__(directory)

        self.expedition = expedition
        self.scheduler = scheduler

        self.batchsize = batchsize
        self.wait_time = 60

        return

    def run(self, builder=None, *args, **kwargs) -> None:
        """"""
        super().run(*args, **kwargs)

        expedition = self.expedition
        scheduler = self.scheduler

        # - read metadata from file or database
        with TinyDB(
            self.directory/f"_{self.scheduler.name}_jobs.json", indent=2
        ) as database:
            queued_jobs = database.search(Query().queued.exists())
        queued_names = [q["gdir"][self.UUIDLEN+1:] for q in queued_jobs]

        size = 1
        for i in range(size):
            uid = str(uuid.uuid1())
            batch_name = f"{self.EXP_INDEX}-{i}"
            job_name = uid + "-" + self.EXP_INDEX + "-" + f"{i}"
            wdir = self.directory / (self.EXP_INDEX + "-" + f"{i}")
            if batch_name in queued_names:
                self._print(f"{job_name} at {self.directory.name} was submitted.")
                continue
            wdir.mkdir(parents=True, exist_ok=True)

            if self.scheduler.name == "local":
                expedition.directory = wdir
                expedition.run()
            else:
                inp_fpath = (wdir/f"exp-{uid}.yaml").absolute()
                exp_params = expedition.as_dict()
                with open(inp_fpath, "w") as fopen:
                    yaml.safe_dump(exp_params, fopen)

                scheduler.job_name = job_name
                scheduler.script = wdir/f"{self._script_name}-{uid}"
                scheduler.user_commands = "gdp explore {} --wait {}".format(
                    str(inp_fpath), self.wait_time
                )
                scheduler.write()
                if self._submit:
                    self._print(f"{wdir.name}: {scheduler.submit()}")
                else:
                    self._print(f"{wdir.name} waits to submit.")

            # - update database
            with TinyDB(
                self.directory/f"_{self.scheduler.name}_jobs.json", indent=2
            ) as database:
                _ = database.insert(
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
        self._debug(f"~~~{self.__class__.__name__}+inspect")

        running_jobs = self._get_running_jobs()

        with TinyDB(
            self.directory/f"_{self.scheduler.name}_jobs.json", indent=2
        ) as database:
            for job_name in running_jobs:
                doc_data = database.get(Query().gdir == job_name)
                uid = doc_data["uid"]
                wdir_names = doc_data["wdir_names"]

                self.scheduler.job_name = job_name
                self.scheduler.script = self.directory/f"{self._script_name}-{uid}"

                if self.scheduler.is_finished():
                    # -- check if the job finished properly
                    is_finished = False
                    for x in wdir_names:
                        wdir_path = self.directory/x
                        if not wdir_path.exists():
                            break
                        else:
                            self.expedition.directory = wdir_path
                            if not self.expedition.read_convergence():
                                break
                    else:
                        is_finished = True
                    if is_finished:
                        database.update({"finished": True}, doc_ids=[doc_data.doc_id])
                        #self._print(f"{job_name} finished.")
                    else:
                        warnings.warn("Exploration does not support re-submit.", UserWarning)
                        #if resubmit:
                        #    if self.scheduler.name != "local":
                        #        jobid = self.scheduler.submit()
                        #        self._print(f"{job_name} is re-submitted with JOBID {jobid}.")
                        #    else:
                        #        self._print(f"{job_name} tries to re-run.")
                        #        warnings.warn("Local scheduelr does not support re-run.", UserWarning)
                else:
                    self._print(f"{job_name} is running...")

        return
    
    def retrieve(self, include_retrieved: bool=False, *args, **kwargs):
        """"""
        #raise NotImplementedError(f"{self.__class__.__name__}")
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

        workers = []
        if unretrieved_wdirs:
            unretrieved_wdirs = [pathlib.Path(x) for x in unretrieved_wdirs]
            self._debug(f"unretrieved_wdirs: {unretrieved_wdirs}")
            for p in unretrieved_wdirs:
                self.expedition.directory = p
                workers.extend(self.expedition.get_workers())

        with TinyDB(
            self.directory/f"_{self.scheduler.name}_jobs.json", indent=2
        ) as database:
            for job_name in unretrieved_jobs:
                doc_data = database.get(Query().gdir == job_name)
                database.update({"retrieved": True}, doc_ids=[doc_data.doc_id])
        
        return workers


if __name__ == "__main__":
    ...
