#!/usr/bin/env python3
# -*- coding: utf-8 -*

import json
import pathlib
import tempfile
import time
import uuid
from typing import List, Optional, Tuple

import yaml
from ase import Atoms
from ase.io import read, write
from tinydb import Query, TinyDB

from ..computation.driver import AbstractDriver
from ..potential.manager import AbstractPotentialManager
from ..scheduler.local import LocalScheduler
from ..scheduler.scheduler import AbstractScheduler
from ..utils.command import CustomTimer
from .utils import copy_minimal_frames, get_file_md5
from .worker import AbstractWorker

#: Structure ID Key.
STRU_ID_KEY: str = "identifier"

#: Batch ID key used for tracking jobs.
BATCH_ID_KEY: str = "gdir"  # FIXME: change to batch?


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

    def _prepare_batches(self, identifier, structures, potters, drivers):
        """"""
        # try broadcasting structures to drivers
        num_structures = len(structures)
        num_drivers = len(drivers)

        has_broadcast = False
        if num_structures == num_drivers:
            ...
        else:
            if num_structures == 1:
                structures_ = []
                for _ in range(num_drivers):
                    structures_.extend(structures)
                structures = structures_
                num_structures = len(structures)
                has_broadcast = True
            else:
                raise RuntimeError(
                    f"Failed to broadcast {num_structures =} to {num_drivers =}."
                )

        wdir_names = [f"cand{i}" for i in range(num_drivers)]

        starts, ends = self._split_groups(num_structures)

        batches = []
        batch_numbers = []
        for i, (s, e) in enumerate(zip(starts, ends)):
            selected_indices = list(range(s, e))
            batch_numbers.extend([i for _ in selected_indices])
            curr_wdir_names = [wdir_names[x] for x in selected_indices]
            batches.append(
                (
                    selected_indices,
                    curr_wdir_names,
                )
            )

        # save computation inputs for review
        self._write_inputs(identifier, batch_numbers, wdir_names, has_broadcast)

        return batches

    def run(self, structures, batch: Optional[int] = None, *args, **kwargs) -> str:
        """Run computations in batch.

        The structures and the drivers must be one-to-one.

        Args:
            structures: A plain List[Atoms] or a builder.
            batch: batch number.

        Returns:
            The ID of input structures.

        """
        super().run(*args, **kwargs)

        # prepare batch
        identifier, structures = self._preprocess_structures(structures)
        batches = self._prepare_batches(
            identifier, structures, self.potters, self.drivers
        )
        self._print(
            f"num_computations: {len(self.drivers)} num_batches: {len(batches)}"
        )

        # read metadata from file or database
        database = TinyDB(
            self.directory / f"_{self.scheduler.name}_jobs.json", indent=2
        )

        # TODO: The search only works in default_table for now
        #       store each input structures into one different table
        #       datatable = database.table(identifier + "_structures")
        datatable = database

        queued_jobs = datatable.search(Query().queued.exists())
        queued_batch_names = [q[BATCH_ID_KEY][self.UUIDLEN + 1 :] for q in queued_jobs]

        for ib, (
            curr_indices,
            curr_wdirs,
        ) in enumerate(batches):
            # skip submitted jobs
            batch_name = f"batch-{ib}"
            self._print(f"{batch =} {batch_name =} {identifier =}")
            if batch_name in queued_batch_names:
                continue

            # skip batches except for the given one
            if isinstance(batch, int) and ib != batch:
                continue

            self._run_one_batch(
                identifier, datatable, batch_name, curr_indices, curr_wdirs, structures
            )

        database.close()

        return identifier

    def _write_inputs(
        self, identifier: str, batch_numbers, wdir_names, has_broadcast: bool = False
    ):
        """"""
        inp_fpath = self.directory / "_data" / f"inp-{identifier}.json"
        if inp_fpath.exists():
            return

        grid_params = {}
        grid_params["grid"] = []

        for i, (ib, wdir_name, potter, driver) in enumerate(
            zip(batch_numbers, wdir_names, self.potters, self.drivers)
        ):
            if not has_broadcast:
                stru_i = i
            else:
                stru_i = 0
            comput_data = {"batch": ib, "wdir_name": wdir_name}
            comput_data["builder"] = dict(
                method="reader",
                fname=str(
                    (self.directory / "_data" / f"{identifier}.xyz").relative_to(
                        self.directory
                    )
                ),
                index=f"{stru_i}",
            )
            comput_data["computer"] = {}
            comput_data["computer"]["potter"] = potter.as_dict()
            comput_data["computer"]["driver"] = driver.as_dict()
            grid_params["grid"].append(comput_data)

        with open(inp_fpath, "w") as fopen:
            json.dump(grid_params, fopen, indent=2)

        return

    def _run_one_batch(
        self,
        identifier,
        database,
        batch_name: str,
        batch_indices: List[int],
        wdir_names,
        structures,
    ):
        """Run one batch."""
        # ---
        uid = str(uuid.uuid1())
        job_name = uid + "-" + batch_name

        # - TODO: check whether params for scheduler is changed
        user_commands = "gdp compute {} --batch {}\n".format(
            (self.directory / "_data" / f"inp-{identifier}.json").relative_to(
                self.directory
            ),
            batch_name[6:],
        )
        self.scheduler.user_commands = user_commands

        if self.scheduler.name == "local":
            from ..utils.command import run_command

            run_command(self.directory, self.scheduler.user_commands)
        else:
            # - save scheduler file
            self.scheduler.job_name = job_name
            self.scheduler.script = self.directory / f"run-{uid}.script"

            self.scheduler.write()
            if self._submit:
                self._print(f"{self.directory.name} JOBID: {self.scheduler.submit()}")
            else:
                self._print(f"{self.directory.name} waits to submit.")

        # save the information of this batch to the database
        _ = database.insert(
            {
                "uid": uid,
                STRU_ID_KEY: identifier,
                "batch": int(batch_name[6:]),
                BATCH_ID_KEY: job_name,
                "wdir_names": wdir_names,
                "queued": True,
            }
        )

        return

    @staticmethod
    def run_grid_computations_in_command(wdirs, structures, drivers, print_func):
        """"""
        wdir_names = [wdir.name for wdir in wdirs]
        assert len(set(wdir_names)) == len(
            structures
        ), f"Found duplicated wdirs {len(set(wdir_names))} vs. {len(structures)}..."

        with CustomTimer(name="run-driver", func=print_func):
            for wdir, atoms, driver in zip(wdirs, structures, drivers):
                driver.directory = wdir
                print_func(
                    f"{time.asctime( time.localtime(time.time()) )} {driver.directory.name} is running..."
                )
                driver.reset()
                driver.run(atoms, read_ckpt=True, extra_info=None)

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
            doc_data = database.get(Query()[BATCH_ID_KEY] == job_name)
            identifier = doc_data[STRU_ID_KEY]
            uid = doc_data["uid"]
            wdir_names = doc_data["wdir_names"]

            user_commands = "gdp compute {} --batch {}\n".format(
                (self.directory / "_data" / f"inp-{identifier}.json").relative_to(
                    self.directory
                ),
                doc_data["batch"],
            )
            self.scheduler.user_commands = user_commands

            self.scheduler.job_name = job_name
            self.scheduler.script = self.directory / f"run-{uid}.script"

            if self.scheduler.is_finished():
                # check if the job finished properly
                is_finished = False
                wdir_existence = [(self.directory / x).exists() for x in wdir_names]
                if all(wdir_existence):
                    # FIXME: Assume driver ID from the dir name
                    #        and drivers do not change
                    #        it is better recover the driver from a DB
                    # See also in retrieve!!
                    for wdir_name in wdir_names:
                        curr_wdir = self.directory / wdir_name
                        driver_id = int(curr_wdir.name[4:])
                        curr_driver = self.drivers[driver_id]
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
                        if self.scheduler.name != "local":
                            jobid = self.scheduler.submit()
                            self._print(
                                f"{job_name} is re-submitted with JOBID {jobid}."
                            )
                        else:
                            from ..utils.command import run_command

                            run_command(self.directory, self.scheduler.user_commands)
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
            doc_data = database.get(Query()[BATCH_ID_KEY] == job_name)
            unretrieved_wdirs_.extend(
                (self.directory / w).resolve() for w in doc_data["wdir_names"]
            )
        unretrieved_wdirs = unretrieved_wdirs_

        results = []
        if unretrieved_wdirs:
            unretrieved_wdirs = [pathlib.Path(x) for x in unretrieved_wdirs]
            self._debug(f"unretrieved_wdirs: {unretrieved_wdirs}")
            # FIXME: Assume driver ID from the dir name
            #        and drivers do not change
            #        it is better recover the driver from a DB
            for p in unretrieved_wdirs:
                driver_id = int(p.name[4:])
                driver = self.drivers[driver_id]
                driver.directory = p
                results.append(driver.read_trajectory())

        for job_name in unretrieved_jobs:
            doc_data = database.get(Query()[BATCH_ID_KEY] == job_name)
            database.update({"retrieved": True}, doc_ids=[doc_data.doc_id])

        return results

    def as_dict(self) -> dict:
        """"""
        params = super().as_dict()

        return params


if __name__ == "__main__":
    ...
