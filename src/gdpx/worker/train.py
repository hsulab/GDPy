#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" worker for training potentials
"""

import uuid
import pathlib
from typing import NoReturn, Callable
import warnings
import yaml

import numpy as np

from tinydb import Query, TinyDB

from gdpx.worker.worker import AbstractWorker
from gdpx.potential.trainer import AbstractTrainer
from gdpx.utils.command import run_command


class TrainerBasedWorker(AbstractWorker):

    TRAIN_PREFIX: str = "m"

    _print: Callable = print
    _debug: Callable = print

    def __init__(self, trainer: AbstractTrainer, scheduler, directory=None, *args, **kwargs) -> NoReturn:
        """"""
        super().__init__(directory)

        self.trainer = trainer
        self.scheduler = scheduler

        return
    
    def run(self, dataset, size: int=1, init_models=None, *args, **kwargs) -> None:
        """"""
        super().run(*args, **kwargs)
        assert len(init_models) == size, "The number of init models is inconsistent with size."

        trainer = self.trainer
        scheduler = self.scheduler

        # - read metadata from file or database
        with TinyDB(
            self.directory/f"_{self.scheduler.name}_jobs.json", indent=2
        ) as database:
            queued_jobs = database.search(Query().queued.exists())
        queued_names = [q["gdir"][self.UUIDLEN+1:] for q in queued_jobs]

        # - local mode
        for i in range(size):
            uid = str(uuid.uuid1())
            batch_name = f"{self.TRAIN_PREFIX}{i}"
            job_name = uid + "-" + batch_name
            wdir = self.directory / f"{self.TRAIN_PREFIX}{i}"
            if batch_name in queued_names:
                self._print(f"{job_name} at {self.directory.name} was submitted.")
                continue
            wdir.mkdir(parents=True, exist_ok=True)

            if self.scheduler.name == "local":
                trainer.directory = wdir
                trainer.train(dataset, init_model=init_models[i])
            else:
                # - save trainer file
                trainer_params = {}
                trainer_params["trainer"] = trainer.as_dict()
                # TODO: we set a random seed for each trainer
                #       as a committee will be trained
                trainer_params["trainer"]["random_seed"] = np.random.randint(0, 10000)

                trainer_params["init_model"] = init_models[i]

                trainer_params["dataset"] = dataset.as_dict()
                with open(wdir/"trainer.yaml", "w") as fopen:
                    yaml.dump(trainer_params, fopen)

                scheduler.job_name = job_name
                scheduler.script = wdir / "train.script"
                scheduler.user_commands = "gdp train {}\n".format(str((wdir/"trainer.yaml").resolve()))
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
        self._debug(f"@@@{self.__class__.__name__}+inspect")

        running_jobs = self._get_running_jobs()

        with TinyDB(
            self.directory/f"_{self.scheduler.name}_jobs.json", indent=2
        ) as database:
            for job_name in running_jobs:
                doc_data = database.get(Query().gdir == job_name)
                uid = doc_data["uid"]
                wdir_names = doc_data["wdir_names"]

                self.scheduler.job_name = job_name
                self.scheduler.script = self.directory/"train.script"

                if self.scheduler.is_finished():
                    # -- check if the job finished properly
                    is_finished = False
                    for x in wdir_names:
                        wdir_path = self.directory/x
                        if not wdir_path.exists():
                            break
                        else:
                            self.trainer.directory = wdir_path
                            if not self.trainer.read_convergence():
                                break
                    else:
                        is_finished = True
                    if is_finished:
                        database.update({"finished": True}, doc_ids=[doc_data.doc_id])
                    else:
                        warnings.warn("Trainer does not support re-submit.", UserWarning)
                        #if resubmit:
                        #    if self.scheduler.name != "local":
                        #        jobid = self.scheduler.submit()
                        #        self._print(f"{job_name} is re-submitted with JOBID {jobid}.")
                        #    else:
                        #        warnings.warn("Local scheduler does not support re-submit.", UserWarning)
                else:
                    self._print(f"{job_name} is running...")

        return
    
    def retrieve(self, include_retrieved: bool=False, *args, **kwargs):
        """Retrieve training results.

        """
        self.inspect(*args, **kwargs)
        self._debug(f"@@@{self.__class__.__name__}+retrieve")

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
            #print("unretrieved_wdirs: ", unretrieved_wdirs)
            for p in unretrieved_wdirs:
                self.trainer.directory = p
                results.append(self.trainer.freeze())

        with TinyDB(
            self.directory/f"_{self.scheduler.name}_jobs.json", indent=2
        ) as database:
            for job_name in unretrieved_jobs:
                doc_data = database.get(Query().gdir == job_name)
                database.update({"retrieved": True}, doc_ids=[doc_data.doc_id])

        return results
    
    def _read_results(self, gdirs, *args, **kwargs):
        """"""
        return


if __name__ == "__main__":
    ...