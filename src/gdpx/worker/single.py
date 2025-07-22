#!/usr/bin/env python3
# -*- coding: utf-8 -*


import functools
import gzip
import io
import json
import pathlib
import shutil
import tarfile
import time
import uuid
from typing import Optional, Union

import yaml
from ase import Atoms
from ase.io import read, write
from tinydb import Query, TinyDB

from gdpx.computation.driver import BaseDriver
from gdpx.utils.profiler import CustomTimer

from .worker import BaseWorker


def run_computation_in_commandline(
    structure: Atoms,
    driver: BaseDriver,
    dirname: str,
    directory: pathlib.Path,
    share_wdir: bool = False,
    print_func=print,
) -> None:
    """"""
    if not share_wdir:
        with CustomTimer(name="run-driver", func=print_func):
            driver.directory = directory / dirname
            print_func(
                f"{time.asctime( time.localtime(time.time()) )} {dirname} {driver.directory.name} is running..."
            )
            driver.reset()
            driver.run(structure, read_ckpt=True)
    else:
        # run spc calculations in a shared directory
        (directory / "_data").mkdir(parents=True, exist_ok=True)
        cache_fpath = directory / "_data" / "cache.xyz"
        with CustomTimer(name="run-driver", func=print_func):
            driver.directory = directory / "_shared"
            print_func(
                f"{time.asctime( time.localtime(time.time()) )} {dirname} {driver.directory.name} is running..."
            )
            driver.reset()
            driver.run(structure, read_ckpt=False)
            new_atoms = driver.read_trajectory()[-1]
            new_atoms.info["wdir"] = dirname
            write(cache_fpath, new_atoms, append=True)

    return


class SingleWorker(BaseWorker):

    #: Prefix of the computation folder.
    COMP_PREFIX: str = "cand"

    #: How to retrieve computation results, which should be `single` or `all`.
    _retrieve_mode: str = "single"

    def __init__(self, potter, driver, scheduler, directory="./") -> None:
        """"""
        super().__init__(directory=directory)

        self.potter = potter
        self.driver = driver
        self.scheduler = scheduler

        self._wdir_name = ""

        #: Whether share calc dir for each candidate.
        self._share_wdir: bool = False

        #: Whether the worker is spawned.
        self.is_spawned: bool = False

        return

    @staticmethod
    def from_a_worker(worker) -> "SingleWorker":
        """"""
        single_worker = SingleWorker(worker.potter, worker.driver, worker.scheduler, worker.directory)
        single_worker._share_wdir = worker._share_wdir

        return single_worker

    @property
    def wdir_name(self) -> str:
        """"""

        return self._wdir_name

    @wdir_name.setter
    def wdir_name(self, name: str):
        """"""
        self._wdir_name = name

        return

    def run(self, builder, *args, **kwargs):
        """This worker accepts only a single structure."""
        super().run(*args, **kwargs)

        # Prepare batches
        if isinstance(builder, list):  # assume list[Atoms]
            frames = builder
        else:  # assume it is a builder
            frames = builder.run()

        num_frames = len(frames)
        assert num_frames == 1, f"{self.__class__.__name__} accepts only a single structure."

        # Run computations
        if not self.is_spawned:
            self._run_by_scheduler(
                frames,
            )
        else:
            self._run_by_commandline(
                frames,
            )

        return

    def _run_by_commandline(self, frames: list[Atoms]) -> None:
        """"""
        # Run computations
        run_computation_in_commandline(
            frames[0],
            self.driver,
            self.wdir_name,
            self.directory,
            share_wdir=self._share_wdir,
            print_func=self._print,
        )

        return

    def _run_by_scheduler(self, frames: list[Atoms]) -> None:
        """"""
        uid = str(uuid.uuid1())
        assert self.wdir_name, "Computation folder is not set."
        wdir = self.directory / self.wdir_name
        job_name = uid + "-" + "single"

        metadata_dirpath = self.directory / "_data"
        metadata_dirpath.mkdir(parents=True, exist_ok=True)

        # save worker input for later review
        worker_input_fpath = metadata_dirpath / f"worker.json"

        if not worker_input_fpath.exists():
            worker_input_dict = {}
            worker_input_dict["use_single"] = True
            worker_input_dict["driver"] = self.driver.as_dict()
            worker_input_dict["potential"] = self.potter.as_dict()

            with open(worker_input_fpath, "w") as fopen:
                json.dump(worker_input_dict, fopen, indent=2)

        # Check the filepath of the input structures
        dataset_path = str((metadata_dirpath / f"_gdp_inp.xyz").resolve())
        write(dataset_path, frames[0])

        # Save this batch job to the database
        with TinyDB(self.directory / f"_{self.scheduler.name}_jobs.json", indent=2) as database:
            _ = database.insert(
                dict(
                    uid=uid,
                    # md5 = identifier,
                    gdir=job_name,
                    # group_number=ig,
                    wdir_names=[str(wdir)],
                    queued=True,
                )
            )

        # Run batch
        self._irun(
            uid=uid,
            identifier="_gdp_inp",
            frames=frames,
        )

        return

    def _irun(self, uid: str, identifier: str, frames: list[Atoms]):
        """"""
        # Use structure-specific input worker file
        worker_input_fpath = str((self.directory / "_data" / f"worker-{uid}.json").relative_to(self.directory))

        # Check the filepth of the input structures
        dataset_path = str((self.directory / "_data" / f"{identifier}.xyz").relative_to(self.directory))

        # Update scheduler
        jobscript_fname = f"run-{uid}.script"
        self.scheduler.job_name = uid + "-" + "single"
        self.scheduler.script = self.directory / jobscript_fname

        self.scheduler.user_commands = "gdp -p {} compute {} --spawn\n".format(worker_input_fpath, dataset_path)

        # Update function to execute
        func_to_execute = functools.partial(
            run_computation_in_commandline,
            structure=frames[0],
            driver=self.driver,
            dirname=self.wdir_name,
            directory=self.directory,
            share_wdir=self._share_wdir,
            print_func=self._print,
        )

        # TODO: check whether params for scheduler is changed
        self.scheduler.write()
        job_id = self.scheduler.submit(func_to_execute=func_to_execute)
        self._print(f"{self.wdir_name} JOBID: {job_id}")

        return

    def inspect(self, resubmit=False, *args, **kwargs):
        """"""
        super().inspect(*args, **kwargs)

        running_jobs = self._get_running_jobs()  # Always return one job
        self._debug(f"running_jobs: {running_jobs}")

        with TinyDB(self.directory / f"_{self.scheduler.name}_jobs.json", indent=2) as database:
            for job_name in running_jobs:
                doc_data = database.get(Query().gdir == job_name)
                uid = doc_data["uid"]
                wdir_name = pathlib.Path(doc_data["wdir_names"][0]).name

                self.scheduler.job_name = job_name
                self.scheduler.script = self.directory / f"run-{uid}.script"

                if self.scheduler.is_finished():
                    is_finished = False
                    # check if the job finished properly
                    if not self._share_wdir:
                        self.driver.directory = self.directory / wdir_name
                        if self.driver.read_convergence():
                            is_finished = True
                        else:
                            self._print(f"Found unfinished computation at {wdir_name}.")
                    else:
                        cache_fpath = self.directory / "_data" / "cache.xyz"
                        if cache_fpath.exists() and cache_fpath.stat().st_size > 0:
                            cache_atoms = read(cache_fpath, "-1")
                            cache_wdir = cache_atoms.info.get("wdir", "")
                            if cache_wdir == wdir_name:
                                is_finished = True
                            else:
                                self._print(f"Found unfinished computation at {wdir_name}.")
                        else:
                            ...
                    if is_finished:
                        self._print(f"{job_name} is finished...")
                        database.update({"finished": True}, doc_ids=[doc_data.doc_id])
                    else:
                        if resubmit:
                            # jobid = self.scheduler.submit()
                            # self._print(f"{job_name} is re-submitted with JOBID {jobid}.")
                            raise NotImplementedError("Resubmit is not implemented for SingleWorker.")
                else:
                    self._print(f"{job_name} is running...")

        return

    def retrieve(self, include_retrieved: bool = False, use_archive: bool = False, *args, **kwargs):
        """Retrieve training results.

        Args:
            use_archive: Whether archive finished computation folders.

        """
        self.inspect(*args, **kwargs)
        self._print(f"<<-- {self.__class__.__name__}+retrieve -->>")

        unretrieved_wdirs_ = []
        if not include_retrieved:
            unretrieved_jobs = self._get_unretrieved_jobs()
        else:
            unretrieved_jobs = self._get_finished_jobs()

        with TinyDB(self.directory / f"_{self.scheduler.name}_jobs.json", indent=2) as database:
            for job_name in unretrieved_jobs:
                doc_data = database.get(Query().gdir == job_name)
                unretrieved_wdirs_.extend([pathlib.Path(w).resolve() for w in doc_data["wdir_names"]])
            self._debug(f"{unretrieved_wdirs_ = }")
            if self._retrieve_mode == "all":
                unretrieved_wdirs = [p for p in unretrieved_wdirs_]
            elif self._retrieve_mode == "single":
                unretrieved_wdirs = [p for p in unretrieved_wdirs_ if p.name == self.wdir_name]
            else:
                # The retreive mode should be checked before.
                raise Exception(f"Invalid retrieve mode: {self._retrieve_mode}.")

        results = []
        if unretrieved_wdirs:
            if not self._share_wdir:
                # Check if the computation folder exists,
                # and the computation folders should have the same name convention starts with cand!
                existed_wdirs = list([x.resolve() for x in self.directory.glob(f"{self.COMP_PREFIX}*")])
                unretrieved_wdirs = [x for x in unretrieved_wdirs if x in existed_wdirs]
                unretrieved_wdirs = [pathlib.Path(x) for x in unretrieved_wdirs]
                results = self._read_results_from_separate_dirs(unretrieved_wdirs, use_archive=use_archive)
            else:
                unretrieved_wdirs = [pathlib.Path(x) for x in unretrieved_wdirs]
                assert (
                    self._retrieve_mode == "single"
                ), "SingleWorker should only retrieve single computation folder when share_wdir is True."
                cache_fpath = self.directory / "_data" / "cache.xyz"
                cache_atoms = read(cache_fpath, "-1")
                cache_wdir = cache_atoms.info.get("wdir", "")
                unretrieved_wdirname = unretrieved_wdirs[0].name
                assert (
                    cache_wdir == unretrieved_wdirname
                ), f"The unretrieved folder name `{unretrieved_wdirname}` does not match the cache file `{cache_wdir}`."
                results = [[cache_atoms]]
        else:
            ...  # Nothing to retrieve

        with TinyDB(self.directory / f"_{self.scheduler.name}_jobs.json", indent=2) as database:
            for job_name in unretrieved_jobs:
                doc_data = database.get(Query().gdir == job_name)
                database.update({"retrieved": True}, doc_ids=[doc_data.doc_id])

        return results

    def _read_results_from_separate_dirs(self, unretrieved_wdirs: list[pathlib.Path], use_archive: bool = False):
        """"""
        # Find existed computation folders.
        is_archived = False

        archive_path = (self.directory / "cand.tgz").absolute()
        if not archive_path.exists():
            unretrieved_and_unarchived_wdirs = unretrieved_wdirs
            results = self._read_results(
                unretrieved_wdirs,
            )
        else:
            # Find previously archived wdirs
            archived_wdirs = []
            with tarfile.open(archive_path, "r:gz") as tar:
                for tarinfo in tar:
                    if tarinfo.isdir() and tarinfo.name.startswith(self.COMP_PREFIX):
                        archived_wdirs.append(self.directory / tarinfo.name)
                    else:
                        ...
            archived_wdirs = sorted(archived_wdirs, key=lambda x: int(x.name[len(self.COMP_PREFIX) :]))
            self._debug(f"{archived_wdirs = }")
            # TODO: Let driver determines when the computation folder is archived or not.
            # TODO: Deal with a situation where archived and unarchived ones are mixed?
            unretrieved_and_unarchived_wdirs = [x for x in unretrieved_wdirs if x not in archived_wdirs]
            if len(unretrieved_and_unarchived_wdirs) > 0:
                is_archived = False
            else:
                is_archived = True
            if is_archived:
                results = self._read_results(unretrieved_wdirs, archive_path)
            else:
                results = self._read_results(unretrieved_wdirs)

        # Archive results if it has not been done yet.
        if use_archive and not is_archived:
            self._print("archive computation folders...")
            if not archive_path.exists():
                # with tarfile.open(archive_path, "w:gz") as tar:
                #    for w in unretrieved_wdirs:
                #        tar.add(w, arcname=w.name)
                archive_data = io.BytesIO()
                # -- append
                with tarfile.open(fileobj=archive_data, mode="w") as tar:
                    for w in unretrieved_wdirs:
                        self._debug(f"add {w.name} to archive.")
                        tar.add(w, arcname=w.name)
                archive_data.seek(0)
            else:
                # -- load
                archive_data = io.BytesIO()
                with gzip.open(archive_path, "rb") as gzf:
                    archive_data.write(gzf.read())
                archive_data.seek(0)
                # -- append
                with tarfile.open(fileobj=archive_data, mode="a") as tar:
                    for w in unretrieved_and_unarchived_wdirs:
                        self._debug(f"add {w.name} to archive.")
                        tar.add(w, arcname=w.name)
                archive_data.seek(0)
            # -- save archive
            with gzip.open(archive_path, mode="wb", compresslevel=6) as gzf:
                gzf.write(archive_data.read())
            for w in unretrieved_and_unarchived_wdirs:
                shutil.rmtree(w)
        else:
            ...  # Nothing to archive

        return results

    def _read_results(self, unretrieved_wdirs: list[pathlib.Path], archive_path: pathlib.Path = None, *args, **kwargs):
        """"""
        # NOTE: SingleWorker always have one unretrieved directory...
        #       However, the retrieve mode can be set to all to get all computation
        #       results in the current main working directory `self.directory`.
        # TODO: Add joblib!!!
        results = []
        for p in unretrieved_wdirs:
            self.driver.directory = p
            results.append(self.driver.read_trajectory(archive_path=archive_path))

        # TODO: dump results shape? assume the results are 2D
        # if self._retrieve_mode == "all":
        #    dim1 = len(results)
        #    dim2 = max([len(x) for x in results])
        #    size = (dim1, dim2)
        #    markers = []
        #    for i in range(dim1):
        #        maxj = len(results[i])
        #        markers.extend([(i, j) for j in range(maxj)])

        #    #shape_dir = self.directory / "_shape"
        #    shape_dir = self.directory.parent / "_shape"
        #    shape_dir.mkdir(parents=True, exist_ok=True)
        #    np.savetxt(shape_dir / "shape.dat", np.array(size, dtype=np.int32), fmt="%8d")
        #    np.savetxt(shape_dir / "markers.dat", np.array(markers, dtype=np.int32), fmt="%8d")

        return results

    def rewind_to_step(self, step: int):
        """Remove previous computation folders.

        This is used in some expeditions (e.g. MC) as they may restart from
        a checkpoint and the unchecked computation folders will be removed.

        """

        def test_func(wdir_names, step: int) -> bool:
            """"""
            cand_index = int(pathlib.Path(wdir_names[0]).name[4:])

            return cand_index > step

        with TinyDB(self.directory / f"_{self.scheduler.name}_jobs.json", indent=2) as database:
            doc_data = database.search(Query().wdir_names.test(test_func, step))  # type: ignore
            doc_ids = [doc.doc_id for doc in doc_data]
            if doc_ids:
                database.remove(doc_ids=doc_ids)

        return

    def as_dict(self) -> dict:
        """"""
        params = super().as_dict()
        params["use_single"] = True

        return params


if __name__ == "__main__":
    ...
