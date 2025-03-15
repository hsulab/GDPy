#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import functools
import json
import pathlib
import time
import uuid
import warnings
from typing import Optional, Union

from tinydb import Query, TinyDB

from gdpx.expedition.expedition import BaseExpedition
from gdpx.scheduler.scheduler import BaseScheduler

from .worker import BaseWorker

"""Worker that manages expeditions.

Since an expedition is made up of several basic workers, this worker is a monitor that 
tracks its progress.

"""

#: Prefix of the expedition folder name.
EXP_DIR_PREFIX: str = "expedition"


def run_expedition_in_commandline(wdir, expedition, timewait: Optional[float] = None, print_func=print) -> None:
    """"""
    expedition.directory = wdir
    if timewait is not None:
        for _ in range(1000):
            expedition.run()
            if expedition.read_convergence():
                break
            print_func(f"wait {timewait} seconds...")
            time.sleep(timewait)
        else:
            ...
    else:
        expedition.run()

    return


def get_expedition_function(
    expedition, working_directory: pathlib.Path, timewait: Optional[float] = None, print_func=print
):
    """"""
    exp_func = functools.partial(
        run_expedition_in_commandline,
        wdir=working_directory,
        expedition=expedition,
        timewait=timewait,
        print_func=print_func,
    )

    return exp_func


def save_expedition_input_parameters(inp_fpath: pathlib.Path, expedition):
    """"""
    exp_params = expedition.as_dict()
    with open(inp_fpath, "w") as fopen:
        json.dump(exp_params, fopen, indent=2)

    return


class ExpeditionBasedWorker(BaseWorker):

    batchsize: int = 1

    _script_name: str = "run.script"

    def __init__(
        self,
        expedition: Union[BaseExpedition, list[BaseExpedition]],
        scheduler: BaseScheduler,
        batchsize: int = 1,
        timewait: Optional[float] = 60.0,
        directory: Optional[Union[str, pathlib.Path]] = None,
    ) -> None:
        """"""
        super().__init__(directory=directory)

        self.expedition = expedition
        self.scheduler = scheduler

        self.batchsize = batchsize
        self.timewait = timewait

        if self.batchsize != 1:
            raise Exception("Currently, expedition worker only supports batchsize of 1.")

        return

    def run(self, *args, **kwargs) -> None:
        """"""
        super().run(*args, **kwargs)

        # Read metadata from file or database
        with TinyDB(self.directory / f"_{self.scheduler.name}_jobs.json", indent=2) as database:
            queued_jobs = database.search(Query().queued.exists())
        queued_names = [q["gdir"][self.UUIDLEN + 1 :] for q in queued_jobs]
        queued_uuids = [q["uid"] for q in queued_jobs]

        # Check if input expeditions are consistent with those in the database
        if isinstance(self.expedition, list):
            expeditions = self.expedition
        else:
            expeditions = [self.expedition]

        # Submit jobs
        num_expeditions = len(expeditions)
        for i in range(num_expeditions):
            # Check if the job is already submitted and get a new uuid if not
            batch_name = f"{EXP_DIR_PREFIX}-{i}"
            if batch_name in queued_names:
                uid = queued_uuids[i]
                job_name = uid + "-" + EXP_DIR_PREFIX + "-" + f"{i}"
                self._print(f"{job_name} at {self.directory.name} was submitted.")
                continue
            else:
                uid = str(uuid.uuid1())
                job_name = uid + "-" + EXP_DIR_PREFIX + "-" + f"{i}"

            wdir = self.directory / (EXP_DIR_PREFIX + "-" + f"{i}")
            wdir.mkdir(parents=True, exist_ok=True)

            # Get expedition
            expedition = expeditions[i]

            # Save input file
            metadata_dpath = wdir / "_data"
            metadata_dpath.mkdir(parents=True, exist_ok=True)
            inp_fpath = (metadata_dpath / f"exp-{uid}.json").resolve()
            save_expedition_input_parameters(inp_fpath, expedition)

            # Submit expedition to queue
            exp_func = get_expedition_function(
                expedition=expedition, working_directory=wdir, timewait=self.timewait, print_func=self._print
            )

            self.scheduler.job_name = job_name
            self.scheduler.script = wdir / f"{self._script_name}-{uid}"
            relative_inp_fpath = str(inp_fpath.relative_to(wdir.resolve()))
            batch_index_str = ",".join([str(i)])
            self.scheduler.user_commands = (
                f"gdp explore {relative_inp_fpath} --wait {self.timewait} --spawn {batch_index_str}\n"
            )
            job_status = self.scheduler.submit(func_to_execute=exp_func)
            self._print(f"{wdir.name}: {job_status}")

            # Update database
            with TinyDB(self.directory / f"_{self.scheduler.name}_jobs.json", indent=2) as database:
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

    def inspect(self, resubmit: bool = False, *args, **kwargs):
        """"""
        self._initialise(*args, **kwargs)
        self._debug(f"<<-- {self.__class__.__name__}+inspect -->>")

        if isinstance(self.expedition, list):
            expeditions = self.expedition
        else:
            expeditions = [self.expedition]

        running_jobs = self._get_running_jobs()
        with TinyDB(self.directory / f"_{self.scheduler.name}_jobs.json", indent=2) as database:
            for job_name in running_jobs:
                # Set scheduler information
                doc_data = database.get(Query().gdir == job_name)
                uid = doc_data["uid"]

                # Update scheduler with job information
                self.scheduler.job_name = job_name
                self.scheduler.script = self.directory / f"{self._script_name}-{uid}"

                # Get expedition indices
                wdir_names = doc_data["wdir_names"]

                # We only support one expedition per job for now
                assert len(wdir_names) == 1, f"More than one working directory found for {job_name}."

                if self.scheduler.is_finished():
                    # Check if the job finished properly
                    is_finished = False
                    wdir_existence = [(self.directory / x).exists() for x in wdir_names]
                    nwdir_exists = sum(1 for x in wdir_existence if x)
                    if all(wdir_existence):
                        for wdir_name in wdir_names:
                            exp_index = int(wdir_name[len("expedition-") :])
                            self._print(f"{exp_index=}")
                            wdir_path = self.directory / wdir_name
                            if not wdir_path.exists():
                                break
                            else:
                                expeditions[exp_index].directory = wdir_path
                                if not expeditions[exp_index].read_convergence():
                                    break
                        else:
                            is_finished = True
                    else:
                        self._print(f"NOT all workding directories exist.")
                    self._print(f"progress: {nwdir_exists}/{len(wdir_existence)}")
                    if is_finished:
                        database.update({"finished": True}, doc_ids=[doc_data.doc_id])
                    else:
                        if resubmit:
                            # Get the expedition
                            batch_index = int(wdir_names[0][len("expedition-") :])
                            batch_fpath = self.directory / (f"{EXP_DIR_PREFIX}-{batch_index}")
                            exp_func = get_expedition_function(
                                expedition=expeditions[batch_index],
                                working_directory=batch_fpath,
                                timewait=self.timewait,
                                print_func=self._print,
                            )
                            # Update the scheduler
                            self.scheduler.script = batch_fpath / f"{self._script_name}-{uid}"
                            job_id = self.scheduler.submit(func_to_execute=exp_func)
                            self._print(f"{job_name} is re-submitted with JOBID: {job_id}...")
                        else:
                            self._print(f"Resubmit is disabled.")
                else:
                    self._print(f"{job_name} is running...")

        return

    def retrieve(self, include_retrieved: bool = False, *args, **kwargs):
        """"""
        # raise NotImplementedError(f"{self.__class__.__name__}")
        self.inspect(*args, **kwargs)
        self._debug(f"<<-- {self.__class__.__name__}+retrieve -->>")

        unretrieved_wdirs_ = []
        if not include_retrieved:
            unretrieved_jobs = self._get_unretrieved_jobs()
        else:
            unretrieved_jobs = self._get_finished_jobs()

        with TinyDB(self.directory / f"_{self.scheduler.name}_jobs.json", indent=2) as database:
            for job_name in unretrieved_jobs:
                doc_data = database.get(Query().gdir == job_name)
                unretrieved_wdirs_.extend((self.directory / w).resolve() for w in doc_data["wdir_names"])
            unretrieved_wdirs = unretrieved_wdirs_

        # Get expeditions
        if isinstance(self.expedition, list):
            expeditions = self.expedition
        else:
            expeditions = [self.expedition]

        workers = []
        if unretrieved_wdirs:
            unretrieved_wdirs = [pathlib.Path(x) for x in unretrieved_wdirs]
            self._debug(f"unretrieved_wdirs: {unretrieved_wdirs}")
            for p in unretrieved_wdirs:
                exp_index = int(p.name[len("expedition-") :])
                expedition = expeditions[exp_index]
                expedition.directory = p
                workers.extend(expedition.get_workers())

        with TinyDB(self.directory / f"_{self.scheduler.name}_jobs.json", indent=2) as database:
            for job_name in unretrieved_jobs:
                doc_data = database.get(Query().gdir == job_name)
                database.update({"retrieved": True}, doc_ids=[doc_data.doc_id])

        return workers


if __name__ == "__main__":
    ...
