#!/usr/bin/env python3
# -*- coding: utf-8 -*


import copy
import functools
import itertools
import json
import pathlib
import tempfile
import time
import uuid
from typing import Optional, Union

import numpy as np
import omegaconf
from ase import Atoms
from ase.io import read, write
from joblib import Parallel, delayed
from tinydb import Query, TinyDB

from gdpx.data.array import AtomsNDArray
from gdpx.potential.manager import BasePotentialManager
from gdpx.reactor.reactor import BaseReactor
from gdpx.utils.profiler import CustomTimer

from .utils import copy_minimal_frames, get_file_md5, read_cache_info, split_batches
from .worker import BaseWorker


def run_reaction_in_commandline(
    identifier: str,
    structures,
    structure_indices,
    reaction_dirnames: list[str],
    driver: BaseReactor,
    directory: pathlib.Path,
    print_func=print,
) -> None:
    """"""
    # Check machine-specific prefix
    machine_prefix_fpath = directory / "_data" / f"MACHINE_{identifier}"
    if machine_prefix_fpath.exists():
        with open(machine_prefix_fpath, "r") as fopen:
            machine_prefix = "".join(fopen.readlines()).strip()
    else:
        machine_prefix = ""

    # Run reactions
    with CustomTimer(name="run-reactor", func=print_func):
        prev_machine_prefix = driver.setting.machine_prefix
        if machine_prefix:
            driver.setting.machine_prefix = machine_prefix
        for group_indices, dirname in zip(structure_indices, reaction_dirnames):
            driver.directory = directory / dirname
            print_func(
                f"{time.asctime( time.localtime(time.time()) )} {str(dirname)} {driver.directory.name} is running..."
            )
            driver.reset()
            driver.run([structures[i] for i in group_indices], read_ckpt=True)

        # restore machine prefix
        driver.setting.machine_prefix = prev_machine_prefix

    return


class ReactorBasedWorker(BaseWorker):
    """Monitor driver-based jobs."""

    #: The prefix of the computation directory for each reaction.
    wdir_prefix: str = "pair"

    #: Whether the worker is spawned.
    is_spawned: bool = False

    def __init__(
        self,
        potter,
        driver: BaseReactor,
        scheduler=None,
        *args,
        **kwargs,
    ):
        """"""
        super().__init__(*args, **kwargs)

        assert isinstance(potter, BasePotentialManager)

        self.potter = potter
        self.driver = driver
        self.scheduler = scheduler

        return

    def _preprocess(self, structures: Union[list[Atoms], AtomsNDArray]):
        """"""
        # The metadata directory
        metadata_dpath = self.directory / "_data"
        metadata_dpath.mkdir(exist_ok=True)

        # Group structures into pairs or bands
        if isinstance(structures, list):
            # Get reaction groups from atoms.info["rxn_grp"]
            reaction_groups = []
            for atoms in structures:
                rxn_grp = atoms.info.get("rxn_grp", -1)
                if isinstance(rxn_grp, np.ndarray):
                    rxn_grp = rxn_grp.tolist()  # Convert numpy array to list
                else:
                    rxn_grp = [rxn_grp]  # If there is only one number, it will be np.int64
                reaction_groups.append(rxn_grp)
            rxn_indices = list(itertools.chain(*reaction_groups))
            have_only_one_reaction = all([x == -1 for x in rxn_indices])
            if not have_only_one_reaction:
                min_idx, max_idx = min(rxn_indices), max(rxn_indices)
                num_reactions = max_idx - min_idx + 1
                # Group structures by their reaction group
                groups = [[] for _ in range(num_reactions)]
                for i, rxn_grp in enumerate(reaction_groups):
                    for rg in rxn_grp:
                        groups[rg - min_idx].append(i)
            else:
                # For compatibility, the input are just images for one neb calculation
                groups = [list(range(len(structures)))]
        elif isinstance(structures, AtomsNDArray):
            # if structures.ndim == 3:  # from extract
            #     assert structures.shape[0] == 2, "Structures must have a shape of (2, ?, ?)."
            #     pairs = []
            #     for p in structures:
            #         p = [[a for a in s if a is not None][-1] for s in p]
            #         pairs.append(p)
            #     pairs = list(zip(pairs[0], pairs[1]))
            # elif structures.ndim == 2:  # from extract
            #     pairs = list(zip(structures[0], structures[1]))
            #     # raise RuntimeError()
            # else:
            #     pairs = []
            #     raise RuntimeError()
            raise NotImplementedError("AtomsNDArray is not supported yet.")
        else:
            raise Exception(f"Unsupported input structure type `{type(structures)}`.")

        # Compare the input structures
        frames, curr_info = copy_minimal_frames(structures)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".xyz") as tmp:
            write(
                tmp.name,
                frames,
                columns=["symbols", "positions", "move_mask"],
            )

            with open(tmp.name, "rb") as fopen:
                identifier = get_file_md5(fopen)

        _info_data = read_cache_info(self.directory, self.UUIDLEN)

        cache_fname = f"{identifier}.xyz"
        if (metadata_dpath / cache_fname).exists():
            self._print(f"Found file with md5 {identifier}")
            self._info_data = _info_data
            start_confid = 0
            for x in self._info_data:
                if x[1] == identifier:
                    break
                start_confid += 1
            if (metadata_dpath / f"{identifier}_reactions.json").exists():
                with open(metadata_dpath / f"{identifier}_reactions.json", "r") as fopen:
                    groups = json.load(fopen)
        else:
            # Save structures
            write(
                metadata_dpath / cache_fname,
                frames,
            )
            # Save current atoms.info and append curr_info to _info_data
            start_confid = len(_info_data)
            content = "{:<12s}  {:<32s}  {:<12s}  {:<12s}  {:<s}\n".format("#id", "MD5", "confid", "step", "wdir")
            for i, (confid, step, wdir) in enumerate(curr_info):
                line = "{:<12d}  {:<32s}  {:<12d}  {:<12d}  {:<s}\n".format(
                    i + start_confid, identifier, confid, step, wdir
                )
                content += line
                _info_data.append(line.strip().split())
            self._info_data = _info_data
            with open(metadata_dpath / f"{identifier}_info.txt", "w") as fopen:
                fopen.write(content)
            # Save reaction groups
            with open(metadata_dpath / f"{identifier}_reactions.json", "w") as fopen:
                json.dump(groups, fopen, indent=2)

        # Some reactors need energies for IS and FS...
        for grp in groups:
            num_frames_in_group = len(grp)
            if num_frames_in_group >= 2:
                try:
                    ene = frames[grp[0]].get_potential_energy()
                    frames[grp[0]].info["energy"] = ene
                except:
                    ...
                try:
                    ene = frames[grp[-1]].get_potential_energy()
                    frames[grp[-1]].info["energy"] = ene
                except:
                    ...

        return identifier, frames, groups

    def prepare_batches(self, structures):
        """"""
        identifier, frames, groups = self._preprocess(structures)

        num_reactions = len(groups)

        # Overwrite batchsize if share_wdir is used or the scheduler is local
        overwrite_batchsize = False
        if self.scheduler.name == "local":
            overwrite_batchsize = True

        if overwrite_batchsize:
            self._print(f"Overwrites batchsize to {num_reactions=} as it uses local scheduler.")
            batchsize = num_reactions
        else:
            batchsize = self.batchsize

        wdirs = [f"{self.wdir_prefix}{i}" for i in range(num_reactions)]

        # Split structures into different batches
        starts, ends = split_batches(num_reactions, batchsize)

        batches = []
        for _, (s, e) in enumerate(zip(starts, ends)):
            batch_indices = range(s, e)
            batch_dirnames = [wdirs[x] for x in batch_indices]
            batch_structure_indices = [groups[x] for x in batch_indices]
            batches.append([batch_dirnames, batch_structure_indices])

        return identifier, frames, batches

    def run(self, structures: list[list[Atoms]], *args, **kwargs) -> None:
        """"""
        super().run(*args, **kwargs)

        # Prepare batches
        identifier, frames, batches = self.prepare_batches(structures)

        # Some optional arguments
        target_batch = kwargs.get("batch", None)

        if not self.is_spawned:
            self._run_by_scheduler(
                identifier,
                frames,
                batches,
                target_batch=target_batch,
            )
        else:
            self._run_by_commandline(
                identifier,
                frames,
                batches,
                target_batch=target_batch,
            )

        return

    def _run_by_commandline(self, identifier: str, pairs, batches, target_batch) -> None:
        """"""
        # Load metadata for the target batch
        batch = batches[target_batch]

        # Run reactions
        run_reaction_in_commandline(
            identifier=identifier,
            structures=pairs,
            structure_indices=batch[1],
            reaction_dirnames=batch[0],
            driver=self.driver,
            directory=self.directory,
            print_func=self._print,
        )

        return

    def _run_by_scheduler(self, identifier: str, frames, batches, target_batch: Optional[int] = None) -> None:
        """"""
        # Load metadata for previous submitted batches
        database_path = (self.directory / f"_{self.scheduler.name}_jobs.json").resolve()
        self._print(f"database_path: {database_path.relative_to(pathlib.Path.cwd())}")

        with TinyDB(database_path, indent=2) as database:
            queued_jobs = database.search(Query().queued.exists())
        queued_names = [q["gdir"][self.UUIDLEN + 1 :] for q in queued_jobs]
        queued_input = [q["md5"] for q in queued_jobs]

        for ig, batch in enumerate(batches):
            # Set job name
            batch_name = f"group-{ig}"
            uid = str(uuid.uuid1())
            job_name = uid + "-" + batch_name

            # Check whether the job is submitted
            if batch_name in queued_names and identifier in queued_input:
                self._print(f"{batch_name} at {self.directory.name} was submitted.")
                continue

            # Specify which group this worker is responsible for if not, then skip
            # Skip batch here assures the skipped batches will not recorded and
            # thus will not affect their execution if several batches run at the same time.
            if isinstance(target_batch, int):
                if ig != target_batch:
                    self._print(
                        f"{time.asctime( time.localtime(time.time()) )} {self.driver.directory.name} "
                        + f"batch {ig} is skipped..."
                    )
                    continue
                else:
                    ...
            else:
                ...

            # Save this batch job to the database
            if identifier not in queued_input:
                batch_dirnames = batch[0]
                with TinyDB(database_path, indent=2) as database:
                    _ = database.insert(
                        dict(
                            uid=uid,
                            md5=identifier,
                            gdir=job_name,
                            group_number=ig,
                            wdir_names=batch_dirnames,
                            queued=True,
                        )
                    )
                # save worker input for later review
                worker_input_fpath = self.directory / "_data" / f"worker-{identifier}.json"
                if not worker_input_fpath.exists():
                    # TODO: We make sure the dict is python primitive since they may be
                    #       from session nodes, or we should convert it in operations?
                    worker_input_dict = omegaconf.OmegaConf.create(self.as_dict())
                    worker_input_dict = omegaconf.OmegaConf.to_container(worker_input_dict)
                    with open(worker_input_fpath, "w") as fopen:
                        json.dump(worker_input_dict, fopen, indent=2)
                    with open(self.directory / "_data" / f"MACHINE_{identifier}", "w") as fopen:
                        fopen.write(self.scheduler.machine_prefix)

            # Run batch
            self._irun(
                batch_name=batch_name,
                uid=uid,
                identifier=identifier,
                frames=frames,
                batch=batch,
            )

        return

    def _irun(
        self,
        batch_name: str,
        uid: str,
        identifier: str,
        frames: list[Atoms],
        batch,
    ) -> None:
        """Submit one batch either to the queue or to the commandline."""
        batch_number = int(batch_name.split("-")[-1])

        # Use structure-specific input worker file
        worker_input_fpath = str((self.directory / "_data" / f"worker-{identifier}.json").relative_to(self.directory))

        # Check the filepath of the input structures
        dataset_path = str((self.directory / "_data" / f"{identifier}.xyz").relative_to(self.directory))

        # Update scheduler
        jobscript_fname = f"run-{uid}.script"
        self.scheduler.job_name = uid + "-" + batch_name
        self.scheduler.script = self.directory / jobscript_fname

        self.scheduler.user_commands = "gdp -p {} compute {} --batch {} --spawn\n".format(
            worker_input_fpath,
            dataset_path,
            batch_number,
        )

        # Update function to execute
        func_to_execute = functools.partial(
            run_reaction_in_commandline,
            identifier=identifier,
            structures=frames,
            structure_indices=batch[1],
            reaction_dirnames=batch[0],
            driver=self.driver,
            directory=self.directory,
            print_func=self._print,
        )

        # TODO: check whether params for scheduler is changed
        self.scheduler.write()
        if self._submit:
            job_id = self.scheduler.submit(func_to_execute=func_to_execute)
            self._print(f"{self.directory.name} JOBID: {job_id}")
        else:
            self._print(f"{self.directory.name} waits to submit.")

        return

    def inspect(self, resubmit=False, *args, **kwargs):
        """Check if any job were finished correctly not due to time limit.

        Args:
            resubmit: Check whether submit unfinished jobs.

        """
        self._initialise(*args, **kwargs)
        self._print(f"<<-- {self.__class__.__name__}+inspect -->>")

        running_jobs = self._get_running_jobs()

        with TinyDB(self.directory / f"_{self.scheduler.name}_jobs.json", indent=2) as database:
            for job_name in running_jobs:
                doc_data = database.get(Query().gdir == job_name)
                uid = doc_data["uid"]
                identifier = doc_data["md5"]
                curr_batch = doc_data["group_number"]

                self.scheduler.job_name = job_name
                self.scheduler.script = self.directory / f"run-{uid}.script"

                # Check if the job is still running (or in the queue)
                # True if the task finished correctly not due to time-limit
                if self.scheduler.is_finished():
                    is_finished = False
                    wdir_names = doc_data["wdir_names"]
                    # first a quick check if all wdirs exist
                    wdir_existence = [(self.directory / x).exists() for x in wdir_names]
                    nwdir_exists = sum(1 for x in wdir_existence if x)
                    if all(wdir_existence):
                        for x in wdir_names:
                            curr_wdir = self.directory / x
                            self.driver.directory = curr_wdir
                            if not self.driver.read_convergence():
                                self._print(f"Found unfinished computation at {curr_wdir.name}")
                                break
                        else:
                            is_finished = True
                    else:
                        self._print("NOT ALL wdirs exist.")
                    self._print(f"progress: {nwdir_exists}/{len(wdir_existence)}")
                    if is_finished:
                        self._print(f"{job_name} is finished...")
                        doc_data = database.get(Query().gdir == job_name)
                        database.update({"finished": True}, doc_ids=[doc_data.doc_id])
                    else:
                        if resubmit:
                            self._resubmit_by_scheduler(identifier, curr_batch, job_name)
                else:
                    self._print(f"{job_name} is running...")

        return

    def _resubmit_by_scheduler(self, identifier: str, target_batch: int, job_name: str):
        """Load cache and resubmit the job to the scheduler."""
        frames = read(
            self.directory / "_data" / f"{identifier}.xyz",
            ":",
        )
        cache_identifier, cache_pairs, cache_batches = self.prepare_batches(frames)
        assert cache_identifier == identifier, "Inconsistent identifiers for the input structure."
        batch = cache_batches[target_batch]
        func_to_execute = functools.partial(
            run_reaction_in_commandline,
            identifier=identifier,
            structures=cache_pairs,
            structure_indices=batch[1],
            reaction_dirnames=batch[0],
            driver=self.driver,
            directory=self.directory,
            print_func=self._print,
        )
        job_id = self.scheduler.submit(func_to_execute=func_to_execute)
        self._print(f"{job_name} is re-submitted with JOBID: {job_id}...")

        return

    def retrieve(
        self,
        include_retrieved: bool = False,
        given_wdirs: list[str] = None,
        *args,
        **kwargs,
    ):
        """Read results from wdirs.

        Args:
            include_retrieved: Whether include wdirs that are already retrieved.
                              Otherwise, all finished jobs are included.

        """
        self.inspect(*args, **kwargs)
        self._print(f"<<-- {self.__class__.__name__}+retrieve -->>")

        # NOTE: sometimes retrieve is used without run
        # self._info_data = self._read_cached_info() # update _info_data

        # - check status and get latest results
        unretrieved_wdirs_ = []
        if not include_retrieved:
            unretrieved_jobs = self._get_unretrieved_jobs()
        else:
            unretrieved_jobs = self._get_finished_jobs()

        unretrieved_identifiers = []

        with TinyDB(self.directory / f"_{self.scheduler.name}_jobs.json", indent=2) as database:
            for job_name in unretrieved_jobs:
                doc_data = database.get(Query().gdir == job_name)
                unretrieved_identifiers.append(doc_data["md5"])
                unretrieved_wdirs_.extend(self.directory / w for w in doc_data["wdir_names"])

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
            unretrieved_wdirs = [pathlib.Path(x) for x in unretrieved_wdirs]
            results = self._read_results(unretrieved_wdirs, *args, **kwargs)
        else:
            results = []

        with TinyDB(self.directory / f"_{self.scheduler.name}_jobs.json", indent=2) as database:
            for job_name in unretrieved_jobs:
                doc_data = database.get(Query().gdir == job_name)
                database.update({"retrieved": True}, doc_ids=[doc_data.doc_id])

        return results

    def _read_results(
        self, unretrieved_wdirs: list[pathlib.Path], *args, **kwargs
    ) -> Union[list[Atoms], list[list[Atoms]]]:
        """Read results from calculation directories.

        Args:
            unretrieved_wdirs: Calculation directories.

        """
        with CustomTimer(name="read-results", func=self._print):
            # NOTE: works for vasp, ...
            results_ = Parallel(n_jobs=self.n_jobs)(
                delayed(self._iread_results)(self.driver, wdir, info_data=None) for wdir in unretrieved_wdirs
            )

            # NOTE: Failed Calcution, One fail, traj fails
            # TODO: check failed...
            # results = []
            # for i, traj_frames in enumerate(results_):
            #    # - sift error structures
            #    if traj_frames:
            #        error_info = traj_frames[0].info.get("error", None)
            #        if error_info:
            #            self._print(f"Found failed calculation at {error_info}...")
            #        else:
            #            results.append(traj_frames)
            #    else:
            #        self._print(f"Found empty calculation at {str(self.directory)} with cand{i}...")

            results = results_

            if results:
                self._print(f"new_trajectories: {len(results)} nframes of the first: {len(results[0])}")

        return results

    @staticmethod
    def _iread_results(driver, wdir, info_data: dict = None) -> list[Atoms]:
        """Extract results from a single directory.

        This must be a staticmethod as it may be pickled by joblib for parallel
        running.

        Args:
            wdir: Working directory.

        """
        driver.directory = wdir
        # NOTE: name convention, cand1112_field1112_field1112
        confid_ = int(wdir.name.strip("pair").split("_")[0])  # internal name
        if info_data is not None:
            cache_confid = int(info_data[confid_][2])
            if cache_confid >= 0:
                confid = cache_confid
            else:
                confid = confid_
        else:
            confid = confid_

        # NOTE: always return the entire trajectories
        traj_frames = driver.read_trajectory()
        # for a in traj_frames: # TODO: add to metadata?
        #    a.info["confid"] = confid
        #    a.info["wdir"] = str(wdir.name)

        return traj_frames

    def as_dict(self) -> dict:
        """"""
        worker_params = {}
        worker_params["potter"] = self.potter.as_dict()
        worker_params["driver"] = self.driver.as_dict()
        worker_params["scheduler"] = self.scheduler.as_dict()

        worker_params = copy.deepcopy(worker_params)

        worker_params["batchsize"] = self.batchsize

        return worker_params


if __name__ == "__main__":
    ...
