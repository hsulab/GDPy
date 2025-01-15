#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import copy
import json
import pathlib
import shutil
import tarfile
import tempfile
import time
import uuid
import warnings
from typing import List, NoReturn, Tuple, Union, Optional

import numpy as np
import omegaconf
import yaml
from ase import Atoms
from ase.io import read, write
from joblib import Parallel, delayed
from tinydb import Query, TinyDB

from .. import config
from ..builder.builder import StructureBuilder
from ..computation.driver import AbstractDriver
from ..potential.manager import AbstractPotentialManager
from ..scheduler import LocalScheduler
from ..scheduler.scheduler import AbstractScheduler
from ..utils.command import CustomTimer
from .utils import copy_minimal_frames, get_file_md5
from .worker import AbstractWorker
from ..utils.logio import remove_extra_stream_handlers

"""Monitor computation tasks with Worker.

"""


def compare_atoms(a1, a2):
    """Compare structures according to cell, chemical symbols, and positions.

    Structures will be different if the atom order changes.

    """
    c1 = a1.get_cell(complete=True)
    c2 = a2.get_cell(complete=True)
    if np.sum(c1 - c2) >= 1e-8:
        return False

    s1 = a1.get_chemical_symbols()
    s2 = a2.get_chemical_symbols()
    if s1 != s2:
        return False

    p1 = a1.get_positions()
    p2 = a2.get_positions()
    if np.sum(p1 - p2) >= 1e-8:
        return False

    return True


class DriverBasedWorker(AbstractWorker):
    """Monitor driver-based jobs.

    Lifetime: queued (running) -> finished -> retrieved

    Note:
        The database stores each unique job ID and its working directory.

    """

    #: How many structures performed in one job.
    batchsize: int = 1

    #: Print period of share_wdir calculations.
    print_period: int = 100

    #: Reserved keys in atoms.info by gdp.
    reserved_keys: List["str"] = ["energy", "step", "wdir"]

    #: Whether the worker is spawned.
    is_spawned: bool = False

    #: Attached driver object.
    _driver = None

    #: Whether generate an independant random_seed for each candidate's driver.
    _share_random_seed: bool = False

    #: Whether share calc dir for each candidate. Only for command run and spc.
    _share_wdir: bool = False

    #: Whether retain the info stored in atoms and add to trajectory.
    _retain_info: bool = False

    def __init__(
        self, potter_, driver_=None, scheduler_=Optional[AbstractScheduler], *args, **kwargs
    ):
        """"""
        super().__init__(*args, **kwargs)

        assert isinstance(potter_, AbstractPotentialManager), ""
        self.potter = potter_
        self.driver = driver_

        if scheduler_ is not None:
            self.scheduler = scheduler_ 
        else:
            self.scheduler = LocalScheduler()

        return

    @property
    def driver(self) -> AbstractDriver:
        return self._driver

    @driver.setter
    def driver(self, driver_):
        """"""
        assert isinstance(driver_, AbstractDriver), ""
        # TODO: check driver is consistent with potter
        self._driver = driver_
        return

    def _split_groups(
        self, nframes: int, batchsize: int = 1
    ) -> Tuple[List[int], List[int]]:
        """Split nframes into groups."""
        # - split frames
        self._debug(f"split_groups for {nframes} nframes and {batchsize} batchsize.")
        ngroups = int(np.floor(1.0 * nframes / batchsize))
        group_indices = [0]
        for i in range(ngroups):
            group_indices.append((i + 1) * batchsize)
        if group_indices[-1] != nframes:
            group_indices.append(nframes)
        starts, ends = group_indices[:-1], group_indices[1:]
        assert len(starts) == len(ends), "Inconsistent start and end indices..."
        # group_indices = [f"{s}:{e}" for s, e in zip(starts,ends)]

        return (starts, ends)

    def _read_cached_info(self):
        # - read extra info data
        _info_data = []
        for p in (self.directory / "_data").glob("*_info.txt"):
            identifier = p.name[: self.UUIDLEN]  # MD5
            with open(p, "r") as fopen:
                for line in fopen.readlines():
                    if not line.startswith("#"):
                        _info_data.append(line.strip().split())
        _info_data = sorted(_info_data, key=lambda x: int(x[0]))

        return _info_data

    def _read_cached_xinfo(self):
        # - read extra info data
        info_keys, _info_data = [], []
        for p in (self.directory / "_data").glob("*_xinfo.txt"):
            identifier = p.name[: self.UUIDLEN]  # MD5
            with open(p, "r") as fopen:
                lines = fopen.readlines()
                info_keys = lines[0].split()[1:]
                for line in lines:
                    if not line.startswith("#"):
                        _info_data.append(line.strip().split()[1:])
        # _info_data = sorted(_info_data, key=lambda x: int(x[0]))
        assert info_keys, f"info_keys must not be empty."

        return info_keys, _info_data

    def _preprocess(self, builder, *args, **kwargs):
        """"""
        # - get frames
        frames = []
        if isinstance(builder, StructureBuilder):
            frames = builder.run()
        else:
            assert all(
                isinstance(x, Atoms) for x in frames
            ), "Input should be a list of atoms."
            frames = builder
        prev_frames = frames

        # - check differences of input structures
        processed_dpath = self.directory / "_data"
        processed_dpath.mkdir(exist_ok=True)

        # - NOTE: atoms.info is a dict that does not maintain order
        #         thus, the saved file maybe different
        curr_frames, curr_info = copy_minimal_frames(prev_frames)

        # -- get MD5 of current input structures
        # NOTE: if two jobs start too close,
        #       there may be conflicts in checking structures
        with tempfile.NamedTemporaryFile(mode="w", suffix=".xyz") as tmp:
            write(tmp.name, curr_frames, columns=["symbols", "positions", "move_mask"])

            with open(tmp.name, "rb") as fopen:
                curr_md5 = get_file_md5(fopen)

        # - generate random_seeds
        #   NOTE: no matter the job status is, they are generated to sync rng state
        self._print(f"Driver's random_seed: {self.driver.random_seed}")
        if not self._share_random_seed:
            rng = np.random.Generator(np.random.PCG64(self.driver.random_seed))
            random_seeds = rng.integers(0, 1e8, size=len(curr_frames))
            random_seeds = [int(x) for x in random_seeds]  # We need List[int]!!
        else:
            random_seeds = [self.driver.random_seed] * len(curr_frames)

        # - check info data
        _info_data = self._read_cached_info()

        stored_fname = f"{curr_md5}.xyz"
        if (processed_dpath / stored_fname).exists():
            self._print(f"Found file with md5 {curr_md5}")
            self._info_data = _info_data
            start_confid = 0
            for x in self._info_data:
                if x[1] == curr_md5:
                    break
                start_confid += 1
            # NOTE: compatability
            if len(_info_data[0]) > 5:
                random_seeds = [int(x[-1]) for x in _info_data]
            else:
                ...  # Use generated random seeds
        else:
            if self._retain_info:
                info_keys = []
                for a in prev_frames:
                    info_keys.extend(list(a.info.keys()))
                info_keys = sorted(set(info_keys))
                content = (
                    f'{"#id":<12s}  '
                    + ("{:<24s}  " * len(info_keys)).format(*info_keys)
                    + "\n"
                )
                for i, a in enumerate(prev_frames):
                    line = (
                        f"{i:<24d}  "
                        + "  ".join([f"{str(a.info.get(k)):<24s}" for k in info_keys])
                        + "\n"
                    )
                    content += line
                with open(processed_dpath / f"{curr_md5}_xinfo.txt", "w") as fopen:
                    fopen.write(content)
            # - save structures
            write(
                processed_dpath / stored_fname,
                curr_frames,
                # columns=["symbols", "positions", "momenta", "tags", "move_mask"]
            )
            # - save current atoms.info and append curr_info to _info_data
            start_confid = len(_info_data)
            content = "{:<12s}  {:<32s}  {:<12s}  {:<12s}  {:<s}  {:>24s}\n".format(
                "#id", "MD5", "confid", "step", "wdir", "rs"
            )
            for i, ((confid, step, wdir), rs) in enumerate(
                zip(curr_info, random_seeds)
            ):
                line = "{:<12d}  {:<32s}  {:<12d}  {:<12d}  {:<s}  {:>24d}\n".format(
                    i + start_confid, curr_md5, confid, step, wdir, rs
                )
                content += line
                _info_data.append(line.strip().split())
            self._info_data = _info_data
            with open(processed_dpath / f"{curr_md5}_info.txt", "w") as fopen:
                fopen.write(content)

        return curr_md5, curr_frames, start_confid, random_seeds

    def _prepare_batches(
        self,
        frames: List[Atoms],
        start_confid: int,
        rng_states: Union[List[int], List[dict]],
    ):
        # - check wdir
        num_frames = len(frames)
        # NOTE: get a list even if it only has one structure
        # TODO: a better strategy to deal with wdirs...
        #       conflicts:
        #           merged trajectories from different drivers that all have cand0
        wdirs = []  # [(confid,dynstep), ..., ()]
        for i, atoms in enumerate(frames):
            # -- set wdir
            wdir = "cand{}".format(int(self._info_data[i + start_confid][0]))
            wdirs.append(wdir)
            atoms.info["wdir"] = wdir
        # - check whether each structure has a unique wdir
        assert (
            len(set(wdirs)) == num_frames
        ), f"Found duplicated wdirs {len(set(wdirs))} vs. {num_frames}..."

        # - split structures into different batches
        if self._share_wdir:
            self._print(
                f"Worker overwrites batchsize to {num_frames =} as it uses share_wdir."
            )
            batchsize = num_frames
        else:
            batchsize = self.batchsize

        starts, ends = self._split_groups(num_frames, batchsize)

        batches = []
        for i, (s, e) in enumerate(zip(starts, ends)):
            # - prepare structures and dirnames
            global_indices = range(s, e)
            # NOTE: get a list even if it only has one structure
            cur_frames = [frames[x] for x in global_indices]
            cur_wdirs = [wdirs[x] for x in global_indices]
            curr_rs = [rng_states[x] for x in global_indices]
            for x in cur_frames:
                x.info["group"] = i
            # - check whether each structure has a unique wdir
            assert len(set(cur_wdirs)) == len(
                cur_frames
            ), f"Found duplicated wdirs {len(set(wdirs))} vs. {len(cur_frames)} for group {i}..."

            # - set specific params
            batches.append([global_indices, cur_wdirs, curr_rs])

        return batches

    def run(self, builder=None, rng_states=list(), *args, **kwargs) -> None:
        """Split frames into groups and submit jobs."""
        super().run(*args, **kwargs)

        # - check if the same input structures are provided
        identifier, frames, start_confid, new_rng_states = self._preprocess(builder)
        if rng_states:  # Sometimes we need explicit rng_states as in active learning
            new_rng_states = rng_states
        batches = self._prepare_batches(frames, start_confid, new_rng_states)

        # - read metadata from file or database
        with TinyDB(
            self.directory / f"_{self.scheduler.name}_jobs.json", indent=2
        ) as database:
            queued_jobs = database.search(Query().queued.exists())
        queued_names = [q["gdir"][self.UUIDLEN + 1 :] for q in queued_jobs]
        queued_frames = [q["md5"] for q in queued_jobs]

        for ig, (global_indices, wdirs, rs) in enumerate(batches):
            # - set job name
            batch_name = f"group-{ig}"
            uid = str(uuid.uuid1())
            job_name = uid + "-" + batch_name

            # -- whether store job info
            if self.scheduler.name != "local":
                if batch_name in queued_names and identifier in queued_frames:
                    self._print(f"{batch_name} at {self.directory.name} was submitted.")
                    continue
            else:  # Local Scheduler
                is_resubmit = kwargs.get("resubmit", False)
                if not is_resubmit:
                    # NOTE:  Only re-run computation when resubmit is set
                    if batch_name in queued_names and identifier in queued_frames:
                        self._print(
                            f"{batch_name} at {self.directory.name} was submitted."
                        )
                        continue
                else:
                    ...

            # - specify which group this worker is responsible for
            #   if not, then skip
            #   Skip batch here assures the skipped batches will not recorded and
            #   thus will not affect their execution if several batches run at the same time.
            target_number = kwargs.get("batch", None)
            if isinstance(target_number, int):
                if ig != target_number:
                    with CustomTimer(name="run-driver", func=self._print):
                        self._print(
                            f"{time.asctime( time.localtime(time.time()) )} {self.driver.directory.name} batch {ig} is skipped..."
                        )
                        continue
                else:
                    ...
            else:
                ...

            # - run batch
            # NOTE: For command execution, if computation exits incorrectly,
            #       it will not be recorded. The computation will resume next
            #       time.
            self._irun(
                batch_name,
                uid,
                identifier,
                frames,
                global_indices,
                wdirs,
                rng_states=rs,
                *args,
                **kwargs,
            )

            # - save this batch job to the database
            if identifier not in queued_frames:
                with TinyDB(
                    self.directory / f"_{self.scheduler.name}_jobs.json", indent=2
                ) as database:
                    _ = database.insert(
                        dict(
                            uid=uid,
                            md5=identifier,
                            gdir=job_name,
                            group_number=ig,
                            wdir_names=wdirs,
                            queued=True,
                        )
                    )
                # save worker input for later review
                worker_input_fpath = (
                    self.directory / "_data" / f"worker-{identifier}.json"
                )
                if not worker_input_fpath.exists():
                    # TODO: We make sure the dict is python primitive since they may be 
                    #       from session nodes,
                    #       or we should convert it in operations?
                    worker_input_dict = omegaconf.OmegaConf.create(self.as_dict())
                    worker_input_dict = omegaconf.OmegaConf.to_container(worker_input_dict)
                    with open(worker_input_fpath, "w") as fopen:
                        json.dump(worker_input_dict, fopen, indent=2)

        return

    def _irun(self, *args, **kwargs):
        """"""

        raise NotImplementedError("Function to run a batch of structures is undefined.")

    def inspect(self, resubmit=False, batch=None, *args, **kwargs):
        """Check if any job were finished correctly not due to time limit.

        Args:
            resubmit: Check whether submit unfinished jobs.

        """
        self._initialise(*args, **kwargs)
        self._print(f"<<-- {self.__class__.__name__}+inspect -->>")

        running_jobs = self._get_running_jobs()

        with TinyDB(
            self.directory / f"_{self.scheduler.name}_jobs.json", indent=2
        ) as database:
            for job_name in running_jobs:
                self._debug(f"inspect {job_name}")
                doc_data = database.get(Query().gdir == job_name)
                uid = doc_data["uid"]
                identifier = doc_data["md5"]
                curr_batch = doc_data["group_number"]

                # self.scheduler.set(**{"job-name": job_name})
                self.scheduler.job_name = job_name
                self.scheduler.script = self.directory / f"run-{uid}.script"

                # check if the job is still running (or in the queue) 
                # True if the task finished correctly not due to time-limit
                if self.scheduler.is_finished():
                    is_finished = False  # whether all calculations are correctly finished
                    wdir_names = doc_data["wdir_names"]  # cand0 cand1 ... cand{N-1} cand{N}
                    if hasattr(self.scheduler, "_sync_remote"):
                        self.scheduler._sync_remote(wdir_names=wdir_names)
                    else:
                        ...
                    if not self._share_wdir:
                        # - first a quick check if all wdirs exist
                        wdir_existence = [
                            (self.directory / x).exists() for x in wdir_names
                        ]
                        nwdir_exists = sum(1 for x in wdir_existence if x)
                        if all(wdir_existence):
                            for x in wdir_names:
                                curr_wdir = self.directory / x
                                self.driver.directory = curr_wdir
                                if not self.driver.read_convergence():
                                    self._print(
                                        f"Found unfinished computation at {curr_wdir.name}"
                                    )
                                    break
                            else:
                                is_finished = True
                        else:
                            self._print("NOT ALL wdirs exist.")
                        self._print(f"progress: {nwdir_exists}/{len(wdir_existence)}")
                    else:
                        # We need first check if cache file exists,
                        # sometimes due to unexpected errors (e.g. OOM)
                        # no cache file will be written.
                        is_finished = False
                        cache_fpath = self.directory / "_data" / f"{identifier}_cache.xyz"
                        if cache_fpath.exists():
                            cache_frames = read(cache_fpath, ":")
                            cache_wdirs = [a.info["wdir"] for a in cache_frames]
                            if set(wdir_names) == set(cache_wdirs):
                                is_finished = True
                            else:
                                self._print(
                                    f"Found unfinished computation at cand{len(cache_wdirs)}"
                                )
                        else:
                            ...
                    if is_finished:
                        # -- finished correctly
                        self._print(f"{job_name} is finished...")
                        doc_data = database.get(Query().gdir == job_name)
                        database.update({"finished": True}, doc_ids=[doc_data.doc_id])
                    else:
                        # NOTE: no need to remove unfinished structures
                        #       since the driver would check it
                        if resubmit:
                            if self.scheduler.name != "local":
                                jobid = self.scheduler.submit()
                                self._print(
                                    f"{job_name} is re-submitted with JOBID {jobid}."
                                )
                            else:
                                if self.is_spawned:
                                    # spawned local worker respects batch keyword
                                    if curr_batch == batch:
                                        self._print(
                                            f"{job_name} is re-submitted with local."
                                        )
                                        frames = read(
                                            self.directory / "_data" / f"{identifier}.xyz", ":"
                                        )
                                        self.run(frames, batch=curr_batch, resubmit=True)
                                else:
                                    self._print(
                                        f"{job_name} is re-submitted with local."
                                    )
                                    frames = read(
                                        self.directory / "_data" / f"{identifier}.xyz", ":"
                                    )
                                    self.run(frames, batch=curr_batch, resubmit=True)
                else:
                    self._print(f"{job_name} is running...")

        return

    def retrieve(
        self,
        include_retrieved: bool = False,
        given_wdirs: List[str] = None,
        use_archive: bool = False,
        *args,
        **kwargs,
    ):
        """Read results from wdirs.

        Args:
            include_retrieved: Whether include wdirs that are already retrieved.
                              Otherwise, all finished jobs are included.

        Returns:
            A nested List of Atoms.

        """
        self.inspect(*args, **kwargs)
        self._print(f"<<-- {self.__class__.__name__}+retrieve -->>")

        # NOTE: sometimes retrieve is used without run
        self._info_data = self._read_cached_info()  # update _info_data

        # - check status and get latest results
        unretrieved_wdirs_ = []
        if not include_retrieved:
            unretrieved_jobs = self._get_unretrieved_jobs()
        else:
            unretrieved_jobs = self._get_finished_jobs()

        unretrieved_identifiers = []

        with TinyDB(
            self.directory / f"_{self.scheduler.name}_jobs.json", indent=2
        ) as database:
            for job_name in unretrieved_jobs:
                doc_data = database.get(Query().gdir == job_name)
                unretrieved_identifiers.append(doc_data["md5"])
                unretrieved_wdirs_.extend(
                    self.directory / w for w in doc_data["wdir_names"]
                )

        # - get given wdirs
        unretrieved_wdirs = []
        if given_wdirs is not None:
            for wdir in unretrieved_wdirs_:
                wdir_name = wdir.name
                if wdir_name in given_wdirs:
                    unretrieved_wdirs.append(wdir)
        else:
            unretrieved_wdirs = unretrieved_wdirs_

        # - read results
        if unretrieved_wdirs:
            # - read results
            unretrieved_wdirs = [pathlib.Path(x) for x in unretrieved_wdirs]
            if not self._share_wdir:
                archive_path = (self.directory / "cand.tgz").absolute()
                if not archive_path.exists():  # read unarchived data
                    results = self._read_results(unretrieved_wdirs, *args, **kwargs)
                else:
                    self._print("read archived data...")
                    results = self._read_results(
                        unretrieved_wdirs, archive_path=archive_path, *args, **kwargs
                    )
                # - archive results if it has not been done
                if use_archive and not archive_path.exists():
                    # TODO: Check if all computation folders are valid?
                    self._print("archive computation folders...")
                    with tarfile.open(archive_path, "w:gz", compresslevel=6) as tar:
                        for w in unretrieved_wdirs:
                            tar.add(w, arcname=w.name)
                    for w in unretrieved_wdirs:
                        shutil.rmtree(w)
                else:
                    ...
            else:
                # TODO: deal with traj...
                cache_frames = []
                for identifier in unretrieved_identifiers:
                    cache_frames.extend(
                        read(self.directory / "_data" / f"{identifier}_cache.xyz", ":")
                    )
                wdir_names = [x.name for x in unretrieved_wdirs]
                results_ = [a for a in cache_frames if a.info["wdir"] in wdir_names]
                # - convert to a List[List[Atoms]] as non-shared run
                results_ = [[a] for a in results_]
                # -- re-add info
                if self._retain_info:
                    info_keys, info_data = self._read_cached_xinfo()
                    retained_keys = [
                        k for k in info_keys if k not in self.reserved_keys
                    ]
                    for i, traj_frames in enumerate(results_):
                        retained_dict = {
                            k: v
                            for k, v in zip(info_keys, info_data[i])
                            if k in retained_keys and v is not None
                        }
                        traj_frames[0].info.update(retained_dict)
                results = results_
        else:
            results = []

        with TinyDB(
            self.directory / f"_{self.scheduler.name}_jobs.json", indent=2
        ) as database:
            for job_name in unretrieved_jobs:
                doc_data = database.get(Query().gdir == job_name)
                database.update({"retrieved": True}, doc_ids=[doc_data.doc_id])

        return results

    def _read_results(
        self,
        unretrieved_wdirs: List[pathlib.Path],
        archive_path: pathlib.Path = None,
        *args,
        **kwargs,
    ) -> Union[List[Atoms], List[List[Atoms]]]:
        """Read results from calculation directories.

        Args:
            unretrieved_wdirs: Calculation directories.

        """
        with CustomTimer(name="read-results", func=self._print):
            # NOTE: works for vasp, ...
            results_ = Parallel(n_jobs=self.n_jobs)(
                delayed(self._iread_results)(
                    self.driver,
                    wdir,
                    info_data=self._info_data,
                    archive_path=archive_path,
                )
                for wdir in unretrieved_wdirs
            )

            # -- re-add info
            if self._retain_info:
                info_keys, info_data = self._read_cached_xinfo()
                retained_keys = [k for k in info_keys if k not in self.reserved_keys]
                for i, traj_frames in enumerate(results_):
                    retained_dict = {
                        k: v
                        for k, v in zip(info_keys, info_data[i])
                        if k in retained_keys and v is not None
                    }
                    traj_frames[0].info.update(retained_dict)

            # NOTE: Failed Calcution, One fail, traj fails
            results = []
            for i, traj_frames in enumerate(results_):
                # - sift error structures
                if traj_frames:
                    results.append(traj_frames)
                else:
                    self._print(
                        f"Found empty calculation at {str(self.directory)} with cand{i}..."
                    )

            if results:
                self._print(
                    f"new_trajectories: {len(results)} nframes of the first: {len(results[0])}"
                )

        return results

    @staticmethod
    def _iread_results(
        driver, wdir, info_data: dict = None, archive_path: pathlib.Path = None
    ) -> List[Atoms]:
        """Extract results from a single directory.

        This must be a staticmethod as it may be pickled by joblib for parallel
        running.

        Args:
            wdir: Working directory.

        """
        driver.directory = wdir
        # NOTE: name convention, cand1112_field1112_field1112
        confid_ = int(wdir.name.strip("cand").split("_")[0])  # internal name
        if info_data is not None:
            cache_confid = int(info_data[confid_][2])
            if cache_confid >= 0:
                confid = cache_confid
            else:
                confid = confid_
        else:
            confid = confid_

        # NOTE: always return the entire trajectories
        traj_frames = driver.read_trajectory(
            add_step_info=True, archive_path=archive_path
        )
        for a in traj_frames:
            a.info["confid"] = confid
            a.info["wdir"] = str(wdir.name)

        return traj_frames

    def _write_worker_inputs(self, uid: str):
        """"""
        worker_params = {}
        worker_params["driver"] = self.driver.as_dict()
        worker_params["potential"] = self.potter.as_dict()
        worker_params["batchsize"] = self.batchsize
        worker_params["share_wdir"] = self._share_wdir
        worker_params["retain_info"] = self._retain_info

        with open(self.directory / f"worker-{uid}.yaml", "w") as fopen:
            yaml.dump(worker_params, fopen)

        # TODO: MACHINE file will be overwritten by different batches
        #       even though they are the same.
        with open(self.directory/f"MACHINE", "w") as fopen:
            fopen.write(self.scheduler.machine_prefix)

        return

    def as_dict(self) -> dict:
        """"""
        worker_params = super().as_dict()
        worker_params["batchsize"] = self.batchsize

        return worker_params


class QueueDriverBasedWorker(DriverBasedWorker):

    def _irun(
        self,
        batch_name: str,
        uid: str,
        identifier: str,
        frames: List[Atoms],
        curr_indices: List[int],
        curr_wdirs: List[Union[str, pathlib.Path]],
        rng_states: Union[List[int], List[dict]],
        *args,
        **kwargs,
    ) -> None:
        """"""
        batch_number = int(batch_name.split("-")[-1])

        # - save worker file
        self._write_worker_inputs(uid=uid)

        # - save structures
        dataset_path = str((self.directory / "_data" / f"{identifier}.xyz").relative_to(self.directory))

        # - save scheduler file
        jobscript_fname = f"run-{uid}.script"
        self.scheduler.job_name = uid + "-" + batch_name
        self.scheduler.script = self.directory / jobscript_fname

        self.scheduler.user_commands = "gdp -p {} compute {} --batch {} --spawn\n".format(
            (self.directory / f"worker-{uid}.yaml").name,
            # (self.directory/structure_fname).name
            dataset_path,
            batch_number,
        )

        # - TODO: check whether params for scheduler is changed
        self.scheduler.write()
        if self._submit:
            self._print(f"{self.directory.name} JOBID: {self.scheduler.submit()}")
        else:
            self._print(f"{self.directory.name} waits to submit.")

        return


class CommandDriverBasedWorker(DriverBasedWorker):

    def _irun(
        self,
        batch_name: str,
        uid: str,
        identifier: str,
        frames: List[Atoms],
        curr_indices: List[int],
        curr_wdirs: List[Union[str, pathlib.Path]],
        rng_states: Union[List[int], List[dict]],
        *args,
        **kwargs,
    ) -> None:
        """Run calculations directly in the command line.

        Local execution supports a compact mode as structures will reuse the
        calculation working directory.

        """
        batch_number = int(batch_name.split("-")[-1])

        # - get structures
        curr_frames = [frames[i] for i in curr_indices]

        machine_prefix = ""
        if (self.directory/"MACHINE").exists():
            with open(self.directory/"MACHINE", "r") as fopen:
                machine_prefix = "".join(fopen.readlines()).strip()

        # - run calculations
        with CustomTimer(name="run-driver", func=self._print):
            prev_machine_prefix = self.driver.setting.machine_prefix
            if machine_prefix:
                self.driver.setting.machine_prefix = machine_prefix
            if not self._share_wdir:
                for wdir, atoms, rs in zip(curr_wdirs, curr_frames, rng_states):
                    remove_extra_stream_handlers()
                    self.driver.directory = self.directory / wdir
                    prev_random_seed = self.driver.random_seed
                    self.driver.set_rng(seed=rs)
                    self._print(
                        f"{time.asctime( time.localtime(time.time()) )} {str(wdir)} {self.driver.directory.name} is running..."
                    )
                    self.driver.reset()
                    self.driver.run(atoms, read_ckpt=True, extra_info=None)
                    self.driver.set_rng(seed=prev_random_seed)
            else:
                # read calculation cache
                cache_fpath = self.directory / "_data" / f"{identifier}_cache.xyz"
                if cache_fpath.exists():
                    cache_frames = read(cache_fpath, ":")
                    cache_wdirs = [a.info["wdir"] for a in cache_frames]
                else:
                    cache_wdirs = []

                num_curr_frames = len(curr_frames)

                # run calculations
                temp_wdir = self.directory / "_shared"
                for i, (wdir, atoms, rs) in enumerate(
                    zip(curr_wdirs, curr_frames, rng_states)
                ):
                    if wdir in cache_wdirs:
                        continue
                    if temp_wdir.exists():
                        shutil.rmtree(temp_wdir)
                    self.driver.directory = temp_wdir
                    if i % self.print_period == 0 or i + 1 == num_curr_frames:
                        self._print(
                            f"{time.asctime( time.localtime(time.time()) )} {str(wdir)} {self.driver.directory.name} is running..."
                        )
                    self.driver.set_rng(seed=rs)
                    self.driver.reset()
                    self.driver.run(atoms, read_ckpt=False, extra_info=dict(wdir=wdir))
                    new_atoms = self.driver.read_trajectory()[-1]
                    new_atoms.info["wdir"] = atoms.info["wdir"]
                    # - save data
                    # TODO: There may have conflicts in write as many groups may run at the same time.
                    #       Add protection to the file.
                    write(
                        self.directory / "_data" / f"{identifier}_cache.xyz",
                        new_atoms,
                        append=True,
                    )
            # restore machine prefix
            self.driver.setting.machine_prefix = prev_machine_prefix

        return


if __name__ == "__main__":
    ...
