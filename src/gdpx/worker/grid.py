#!/usr/bin/env python3
# -*- coding: utf-8 -*

import pathlib
import time
import uuid
import yaml
import tempfile

from typing import Optional, Tuple, List

from tinydb import Query, TinyDB

from ase import Atoms
from ase.io import read, write

from ..computation.driver import AbstractDriver
from ..potential.manager import AbstractPotentialManager
from ..utils.command import CustomTimer
from ..scheduler.scheduler import AbstractScheduler
from ..scheduler.local import LocalScheduler
from .worker import AbstractWorker
from .utils import copy_minimal_frames, get_file_md5


#: Structure ID Key.
STRU_ID_KEY: str = "md5"


class GridDriverBasedWorker(AbstractWorker):

    def __init__(
        self,
        potters: List[AbstractPotentialManager],
        drivers: List[AbstractDriver],
        scheduler: AbstractScheduler = LocalScheduler(),
        directory="./",
        *args,
        **kwargs,
    ) -> None:
        """"""
        super().__init__(directory, *args, **kwargs)

        self.potters = potters

        self.drivers = drivers

        self.scheduler = scheduler

        return

    def _preprocess_structures(self, structures) -> Tuple[str, List[Atoms]]:
        """Preprocess structures."""
        if isinstance(structures, list):  # assume List[Atoms]
            structures = structures
        else:  # assume it is a builder
            structures = structures.run()

        # check differences of input structures
        metadata_dpath = self.directory / "_data"
        metadata_dpath.mkdir(exist_ok=True)

        # NOTE: atoms.info is a dict that does not maintain order
        #       thus, the saved file maybe different
        copied_structures, copied_info = copy_minimal_frames(structures)

        # get MD5 of current input structures
        # NOTE: if two jobs start too close,
        #       there may be conflicts in checking structures
        with tempfile.NamedTemporaryFile(mode="w", suffix=".xyz") as tmp:
            write(
                tmp.name,
                copied_structures,
                columns=["symbols", "positions", "move_mask"],
            )

            with open(tmp.name, "rb") as fopen:
                inp_stru_md5 = get_file_md5(fopen)

        # check
        cache_stru_fname = f"{inp_stru_md5}.xyz"
        if (metadata_dpath / cache_stru_fname).exists():
            ...
        else:
            write(metadata_dpath / cache_stru_fname, copied_structures)

        return inp_stru_md5, copied_structures

    def _spawn_computations(self):
        """"""

        return

    def _prepare_batches(self, structures, potters, drivers):
        """"""
        num_structures = len(structures)
        num_drivers = len(drivers)
        assert num_structures == num_drivers

        wdir_names = [f"cand{i}" for i in range(num_drivers)]

        starts, ends = self._split_groups(num_structures)

        batches = []
        for i, (s, e) in enumerate(zip(starts, ends)):
            selected_indices = range(s, e)
            curr_wdir_names = [wdir_names[x] for x in selected_indices]
            curr_structures = [structures[x] for x in selected_indices]
            curr_potters = [potters[x] for x in selected_indices]
            curr_drivers = [drivers[x] for x in selected_indices]
            assert len(set(curr_wdir_names)) == len(
                curr_structures
            ), f"Found duplicated wdirs {len(set(curr_wdir_names))} vs. {len(curr_structures)} for group {i}..."
            batches.append(
                (curr_wdir_names, curr_structures, curr_potters, curr_drivers)
            )

        return batches

    def run(self, structures, batch: int, *args, **kwargs):
        """Run computations in batch.

        The structure and the driver must be one-to-one

        Args:
            structures: A plain List[Atoms] or a builder.

        """
        super().run(*args, **kwargs)

        # prepare batch
        identifier, structures = self._preprocess_structures(structures)
        batches = self._prepare_batches(structures, self.potters, self.drivers)
        self._print(f"num_computations: {len(structures)} num_batches: {len(batches)}")

        # read metadata from file or database
        with TinyDB(
            self.directory / f"_{self.scheduler.name}_jobs.json", indent=2
        ) as database:
            queued_jobs = database.search(Query().queued.exists())
        queued_names = [q["gdir"][self.UUIDLEN + 1 :] for q in queued_jobs]
        queued_structures = [q[STRU_ID_KEY] for q in queued_jobs]

        for ib, (curr_wdirs, curr_structures, curr_potters, curr_drivers) in enumerate(
            batches
        ):
            # skip submitted jobs
            batch_name = f"batch-{ib}"
            self._print(f"{batch =} {batch_name =} {identifier =}")
            if identifier in queued_structures and batch_name in queued_names:
                if self.scheduler.name != "local":
                    continue
                else:
                    ...
            else:
                ...

            # skip batches except for the given one
            if isinstance(batch, int) and ib != batch:
                continue

            self._run_one_batch(
                batch_name, identifier, curr_wdirs, curr_structures, curr_potters, curr_drivers
            )

        return

    def _run_one_batch(self, batch_name: str, identifier, wdir_names, structures, potters, drivers):
        """Run one batch."""
        # ---
        uid = str(uuid.uuid1())
        job_name = uid + "-" + batch_name

        scheduler = self.scheduler
        if scheduler.name == "local":
            with CustomTimer(name="run-driver", func=self._print):
                for wdir_name, atoms, driver in zip(wdir_names, structures, drivers):
                    driver.directory = self.directory / wdir_name
                    self._print(
                        f"{time.asctime( time.localtime(time.time()) )} {wdir_name} {driver.directory.name} is running..."
                    )
                    driver.reset()
                    driver.run(atoms, read_ckpt=True, extra_info=None)
        else:
            worker_params = {}
            worker_params["use_single"] = True
            worker_params["driver"] = self.driver.as_dict()
            worker_params["potential"] = self.potter.as_dict()

            with open(self.directory / f"worker-{uid}.yaml", "w") as fopen:
                yaml.dump(worker_params, fopen)

            # - save structures
            dataset_path = str((self.directory / f"_gdp_inp.xyz").resolve())
            write(dataset_path, frames[0])

            # - save scheduler file
            jobscript_fname = f"run-{uid}.script"
            self.scheduler.job_name = job_name
            self.scheduler.script = self.directory / jobscript_fname

            self.scheduler.user_commands = "gdp -p {} compute {}\n".format(
                (self.directory / f"worker-{uid}.yaml").name, dataset_path
            )

            # - TODO: check whether params for scheduler is changed
            self.scheduler.write()
            if self._submit:
                self._print(f"{self.directory.name} JOBID: {self.scheduler.submit()}")
            else:
                self._print(f"{self.directory.name} waits to submit.")

        # - save this batch job to the database
        with TinyDB(
            self.directory / f"_{self.scheduler.name}_jobs.json", indent=2
        ) as database:
            _ = database.insert(
                dict(
                    uid=uid,
                    md5=identifier,
                    gdir=job_name,
                    wdir_names=wdir_names,
                    queued=True,
                )
            )

        return

    def inspect(self, resubmit=False, *args, **kwargs):
        """"""
        self._initialise(*args, **kwargs)

        running_jobs = self._get_running_jobs()

        with TinyDB(
            self.directory / f"_{self.scheduler.name}_jobs.json", indent=2
        ) as database:
            self._inspect_and_update(
                running_jobs=running_jobs, database=database, resubmit=resubmit
            )

        return

    def _inspect_and_update(self, running_jobs, database, resubmit: bool = True):
        """"""
        for job_name in running_jobs:
            doc_data = database.get(Query().gdir == job_name)
            uid = doc_data["uid"]
            wdir_names = doc_data["wdir_names"]

            self.scheduler.job_name = job_name
            self.scheduler.script = self.directory / f"run-{uid}.script"

            if self.scheduler.is_finished():
                # check if the job finished properly
                is_finished = False
                wdir_existence = [(self.directory / x).exists() for x in wdir_names]
                if all(wdir_existence):
                    # FIXME: use driver id in the db?
                    for i, x in enumerate(wdir_names):
                        curr_wdir = self.directory / x
                        curr_driver = self.drivers[i]
                        curr_driver.directory = curr_wdir
                        if not curr_driver.read_convergence():
                            self._print(
                                f"Found unfinished computation at {curr_wdir.name}"
                            )
                            break
                    else:
                        is_finished = True
                else:
                    self._print("NOT ALL working directories exist.")

                num_wdirs = len(wdir_existence)
                num_wdir_exists = sum(1 for x in wdir_existence if x)
                self._print(f"progress: {num_wdir_exists}/{num_wdirs}.")

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

    def retrieve(self, include_retrieved: bool = False, *args, **kwargs):
        """Retrieve training results."""
        self.inspect(*args, **kwargs)
        self._debug(f"~~~{self.__class__.__name__}+retrieve")

        # check status and get results
        if not include_retrieved:
            unretrieved_jobs = self._get_unretrieved_jobs()
        else:
            unretrieved_jobs = self._get_finished_jobs()

        with TinyDB(
            self.directory / f"_{self.scheduler.name}_jobs.json", indent=2
        ) as database:
            results = self._retrieve_and_update(
                unretrieved_jobs=unretrieved_jobs, database=database
            )

        return results

    def _retrieve_and_update(self, unretrieved_jobs, database):
        """"""
        unretrieved_wdirs_ = []
        for job_name in unretrieved_jobs:
            doc_data = database.get(Query().gdir == job_name)
            unretrieved_wdirs_.extend(
                (self.directory / w).resolve() for w in doc_data["wdir_names"]
            )
        unretrieved_wdirs = unretrieved_wdirs_

        results = []
        if unretrieved_wdirs:
            unretrieved_wdirs = [pathlib.Path(x) for x in unretrieved_wdirs]
            self._debug(f"unretrieved_wdirs: {unretrieved_wdirs}")
            # FIXME: use driver id in the db?
            for i, p in enumerate(unretrieved_wdirs):
                driver = self.drivers[i]
                driver.directory = p
                results.append(driver.read_trajectory())

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
