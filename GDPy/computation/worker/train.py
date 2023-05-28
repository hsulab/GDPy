#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" worker for training potentials
"""

import uuid
import pathlib
from typing import NoReturn, Callable
import yaml

import numpy as np

from tinydb import Query

from GDPy.computation.worker.worker import AbstractWorker
from GDPy.potential.trainer import AbstractTrainer
from GDPy.utils.command import run_command

class TrainWorker(AbstractWorker):

    """ components
        potter + scheduler
    """

    def __init__(self, potter_, scheduler_, directory_=None, *args, **kwargs):
        """"""
        self.potter = potter_
        self.scheduler = scheduler_
        if directory_:
            self.directory = directory_

        return
    
    def run(self, dataset=None, size=1, *args, **kwargs):
        """"""
        super().run(*args, **kwargs)

        potter = self.potter
        scheduler = self.scheduler

        train_dirs = []
        for i in range(size):
            train_dir = self.directory/("m"+str(i))
            if not train_dir.exists():
                train_dir.mkdir()
            train_dirs.append(train_dir)

        # - read metadata from file or database
        queued_jobs = self.database.search(Query().queued.exists())
        queued_names = [q["gdir"][self.UUIDLEN+1:] for q in queued_jobs]
        
        scheduler.user_commands = "\n".join(
            [self.potter.train_command, self.potter.freeze_command]
        )
        for train_dir in train_dirs:
            if train_dir.name in queued_names:
                continue
            # --- write files for training
            potter._make_train_files(dataset, train_dir)
            # ---
            job_name = str(uuid.uuid1()) + "-" + train_dir.name
            #scheduler.set(**{"job-name": job_name})
            scheduler.job_name = job_name
            scheduler.script = train_dir / "train.script"
            if scheduler.name != "local":
                scheduler.write()
                if self._submit:
                    self.logger.info(f"{train_dir.name}: {scheduler.submit()}")
                else:
                    self.logger.info(f"{train_dir.name} waits to submit.")
            else:
                # train directly
                run_command(str(train_dir), self.potter.train_command)
                run_command(str(train_dir), self.potter.freeze_command)
            self.database.insert(dict(gdir=job_name, queued=True))

        return
    
    def _read_results(self, wdirs):
        """freeze models"""
        return

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
    
    def run(self, dataset, size: int=1, init_models=None, *args, **kwargs) -> NoReturn:
        """"""
        super().run(*args, **kwargs)
        assert len(init_models) == size, "The number of init models is inconsistent with size."

        trainer = self.trainer
        scheduler = self.scheduler

        # - read metadata from file or database
        queued_jobs = self.database.search(Query().queued.exists())
        #queued_names = [q["gdir"][self.UUIDLEN+1:] for q in queued_jobs]
        #queued_frames = [q["md5"] for q in queued_jobs]

        # - local mode
        job_name = str(uuid.uuid1()) + "-" + "xxx"
        if self.scheduler.name == "local":
            for i in range(size):
                trainer.directory = self.directory / f"{self.TRAIN_PREFIX}{i}"
                trainer.train(dataset, init_model=init_models[i])
        else:
            for i in range(size):
                uid = str(uuid.uuid1())
                job_name = uid + "-" + f"{self.TRAIN_PREFIX}{i}"
                wdir = self.directory / f"{self.TRAIN_PREFIX}{i}"
                if not wdir.exists():
                    wdir.mkdir(parents=True)
                else:
                    # TODO: better check
                    continue

                # - save trainer file
                trainer_params = {}
                trainer_params["trainer"] = trainer.as_dict()
                # TODO: we set a random seed for each trainer
                #       as a committee will be trained
                trainer_params["trainer"]["random_seed"] = np.random.randint(0,10000)

                trainer_params["init_model"] = init_models[i]

                trainer_params["dataset"] = dataset.as_dict()
                with open(wdir/"trainer.yaml", "w") as fopen:
                    yaml.dump(trainer_params, fopen)

                scheduler.job_name = job_name
                scheduler.script = wdir / "train.script"
                scheduler.user_commands = "gdp newtrain {}\n".format(str((wdir/"trainer.yaml").resolve()))
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
            ...

        return
    
    def inspect(self, resubmit=False, *args, **kwargs):
        """"""
        self._initialise(*args, **kwargs)
        self._debug(f"@@@{self.__class__.__name__}+inspect")

        running_jobs = self._get_running_jobs()
        for job_name in running_jobs:
            doc_data = self.database.get(Query().gdir == job_name)
            uid = doc_data["uid"]

            self.scheduler.job_name = job_name
            self.scheduler.script = self.directory/"train.script"
            
            if self.scheduler.is_finished():
                # -- check if the job finished properly
                #self.trainer.directory = self.directory
                if True:
                    self.database.update({"finished": True}, doc_ids=[doc_data.doc_id])
                else:
                    if resubmit:
                        jobid = self.scheduler.submit()
                        self.logger.info(f"{job_name} is re-submitted with JOBID {jobid}.")
            else:
                self._print(f"{job_name} is running...")

        return
    
    def retrieve(self, ignore_retrieved: bool=False, *args, **kwargs):
        """Retrieve training results.

        Args:
            ignore_retrieved: Ignore the retrieved tag.

        """
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

        results = []
        if unretrieved_wdirs:
            unretrieved_wdirs = [pathlib.Path(x) for x in unretrieved_wdirs]
            print("unretrieved_wdirs: ", unretrieved_wdirs)
            for p in unretrieved_wdirs:
                self.trainer.directory = p
                results.append(self.trainer.freeze())

        for job_name in unretrieved_jobs:
            doc_data = self.database.get(Query().gdir == job_name)
            self.database.update({"retrieved": True}, doc_ids=[doc_data.doc_id])

        return results
    
    def _read_results(self, gdirs, *args, **kwargs):
        """"""
        return


if __name__ == "__main__":
    ...
