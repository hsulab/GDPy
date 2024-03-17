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

from ..data.interface import DatasetVariable
from .worker import AbstractWorker
from ..potential.trainer import AbstractTrainer


class TrainerBasedWorker(AbstractWorker):

    TRAIN_PREFIX: str = "m"

    #: Whether share a dataset when training a group of models.
    _share_dataset: bool = False

    def __init__(
        self,
        trainer: AbstractTrainer,
        scheduler,
        share_dataset: bool = False,
        auto_submit: bool = True,
        directory=None,
        *args,
        **kwargs,
    ) -> None:
        """Initialise a TrainerBasedWorker.

        Args:
            share_dataset: Whether a group of models are traiend on a shared dataset.
            auto_submit:
                Whether submit scheduler jobs automatically. Otherwise, it must be
                submitted manully.

        """
        super().__init__(directory)

        self.trainer = trainer
        self.scheduler = scheduler

        self._share_dataset = share_dataset
        self._submit = auto_submit

        return

    def _get_train_params(
        self,
        trainer: AbstractTrainer,
        dataset,
        init_model,
        use_shared_dataset: bool = False,
    ) -> dict:
        """"""
        trainer_params = {}
        trainer_params["trainer"] = trainer.as_dict()

        # extra params
        trainer_params["trainer"]["share_dataset"] = use_shared_dataset

        # TODO: we set a random seed for each trainer
        #       as a committee will be trained
        #       it changes the trainer's random state as well...
        trainer_random_seed = np.random.randint(0, 10000)
        trainer_params["trainer"]["random_seed"] = trainer_random_seed
        trainer.set_rng(seed=trainer_random_seed)

        trainer_params["init_model"] = init_model
        trainer_params["dataset"] = dataset.as_dict()

        return trainer_params

    def run(self, dataset, size: int = 1, init_models=None, *args, **kwargs) -> None:
        """"""
        super().run(*args, **kwargs)
        if init_models is None:
            init_models = [None for i in range(size)]
        assert (
            len(init_models) == size
        ), "The number of init models is inconsistent with size."

        trainer = self.trainer
        scheduler = self.scheduler

        self._print(f"{trainer.random_seed = }")

        # - dump some metadata in case of resubmit
        meta_path = self._directory / "_meta"
        meta_path.mkdir(parents=True, exist_ok=True)

        meta_params = {}
        meta_params["dataset"] = dataset.as_dict()
        meta_params["size"] = size
        if init_models is not None:  # assume it is a List of path
            meta_params["init_models"] = [str(x) for x in init_models]
        else:
            meta_params["init_models"] = init_models
        with open(meta_path / "info.yaml", "w") as fopen:
            yaml.safe_dump(meta_params, fopen)

        # - check whether share a dataset?
        if size > 1 and self._share_dataset:
            dataset_path = self.directory / "shared_dataset"
            if not dataset_path.exists():
                self._print("prepare a shared dataset...")
                if hasattr(trainer, "_prepare_dataset"):
                    trainer.directory = dataset_path  # NOTE: only for creating dataset
                    dataset = trainer._prepare_dataset(dataset, *args, **kwargs)
                    self._print(f"{dataset =}")
                    with open(dataset_path/"dataset.yaml", "w") as fopen:
                        yaml.safe_dump(dataset.as_dict(), fopen)
                else:
                    self._print(
                        f"{trainer.__class__.__name__} does not support a shared dataset."
                    )
            else:
                # NOTE: sometimes local trainer does not finish,
                #       we need to load the shared dataset
                self._print("shared dataset exists...")
                with open(dataset_path/"dataset.yaml", "r") as fopen:
                    dataset_params = yaml.safe_load(fopen)
                dataset = DatasetVariable(**dataset_params).value
                self._print(f"{dataset =}")
        else:
            self._print("trainers prepare their own datasets...")

        # - read metadata from file or database
        with TinyDB(
            self.directory / f"_{self.scheduler.name}_jobs.json", indent=2
        ) as database:
            queued_jobs = database.search(Query().queued.exists())
        queued_names = [q["gdir"][self.UUIDLEN + 1 :] for q in queued_jobs]

        # - run training
        for i in range(size):
            uid = str(uuid.uuid1())
            batch_name = f"{self.TRAIN_PREFIX}{i}"
            job_name = uid + "-" + batch_name
            wdir = self.directory / f"{self.TRAIN_PREFIX}{i}"
            if batch_name in queued_names:
                self._print(f"{job_name} at {self.directory.name} was submitted.")
                continue
            wdir.mkdir(parents=True, exist_ok=True)

            # - save trainer file for later reference
            # NOTE: YAML accepts only string path
            curr_init_model = init_models[i]
            if curr_init_model is not None:
                curr_init_model = str(curr_init_model)

            trainer_params = self._get_train_params(
                trainer, dataset, curr_init_model, self._share_dataset
            )
            with open(wdir / "trainer.yaml", "w") as fopen:
                yaml.dump(trainer_params, fopen)

            if self.scheduler.name == "local":
                # TODO: async ?
                self._print(f"training model at {wdir.name}...")
                trainer.directory = wdir
                trainer.train(dataset, init_model=curr_init_model)
            else:
                scheduler.job_name = job_name
                scheduler.script = wdir / "train.script"
                scheduler.user_commands = "gdp train {}\n".format(
                    str((wdir / "trainer.yaml").resolve())
                )
                scheduler.write()
                if self._submit:
                    self._print(f"{wdir.name}: {scheduler.submit()}")
                else:
                    self._print(f"{wdir.name} waits to submit.")

            # - update database
            with TinyDB(
                self.directory / f"_{self.scheduler.name}_jobs.json", indent=2
            ) as database:
                _ = database.insert(
                    dict(
                        uid=uid,
                        gdir=job_name,
                        group_number=i,
                        wdir_names=[wdir.name],
                        queued=True,
                    )
                )

        return

    def inspect(self, resubmit=False, *args, **kwargs):
        """"""
        self._initialise(*args, **kwargs)
        self._debug(f"@@@{self.__class__.__name__}+inspect")

        running_jobs = self._get_running_jobs()

        with TinyDB(
            self.directory / f"_{self.scheduler.name}_jobs.json", indent=2
        ) as database:
            for job_name in running_jobs:
                doc_data = database.get(Query().gdir == job_name)
                uid = doc_data["uid"]
                wdir_names = doc_data["wdir_names"]

                self.scheduler.job_name = job_name
                self.scheduler.script = self.directory / "train.script"

                if self.scheduler.is_finished():  # NOTE: scheduler only checks job_name
                    # -- check if the job finished properly
                    is_finished = False
                    for x in wdir_names:
                        wdir_path = self.directory / x
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
                        if resubmit:
                            if self.scheduler.name != "local":
                                self._print(f"RESUBMIT: {str(self.trainer.directory)}")
                                if self._submit:
                                    self.scheduler.script = (
                                        self.trainer.directory / "train.script"
                                    )
                                    jobid = self.scheduler.submit()
                                    self._print(
                                        f"{job_name} is re-submitted with JOBID {jobid}."
                                    )
                                else:
                                    self._print(f"{job_name} waits to submit.")
                            else:
                                # NOTE: If the training runs on local,
                                #       database stores this job only when the training
                                #       is finished.
                                #       The codes below will only be performed when
                                #       self.inspect() is called without self.run() before.
                                # NOTE: Local job will automatically re-run when
                                #       self.run() is called.
                                self._print(f"RESUBMIT: {self.trainer.directory = }")
                                # self.trainer.train(dataset, init_model=init_models[i])
                        else:
                            warnings.warn(
                                "Trainer does not support re-submit.", UserWarning
                            )
                else:
                    self._print(f"{job_name} is running...")

        return

    def retrieve(self, include_retrieved: bool = False, *args, **kwargs):
        """Retrieve training results."""
        self.inspect(*args, **kwargs)
        self._debug(f"@@@{self.__class__.__name__}+retrieve")

        unretrieved_wdirs_ = []
        if not include_retrieved:
            unretrieved_jobs = self._get_unretrieved_jobs()
        else:
            unretrieved_jobs = self._get_finished_jobs()

        with TinyDB(
            self.directory / f"_{self.scheduler.name}_jobs.json", indent=2
        ) as database:
            for job_name in unretrieved_jobs:
                doc_data = database.get(Query().gdir == job_name)
                unretrieved_wdirs_.extend(
                    (self.directory / w).resolve() for w in doc_data["wdir_names"]
                )
        unretrieved_wdirs = unretrieved_wdirs_

        results = []
        if unretrieved_wdirs:
            unretrieved_wdirs = [pathlib.Path(x) for x in unretrieved_wdirs]
            # print("unretrieved_wdirs: ", unretrieved_wdirs)
            for p in unretrieved_wdirs:
                self.trainer.directory = p
                # NOTE: Due to yaml.safe_dump, we require path should be str
                results.append(str(self.trainer.freeze()))

        with TinyDB(
            self.directory / f"_{self.scheduler.name}_jobs.json", indent=2
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
