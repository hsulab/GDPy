#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
import uuid
import pathlib
import shutil
import time
from typing import Tuple, List, NoReturn, Union
import tempfile
import yaml

import numpy as np

from tinydb import Query, TinyDB

from joblib import Parallel, delayed

from ase import Atoms
from ase.io import read, write

from GDPy import config
from GDPy.data.trajectory import Trajectories
from GDPy.potential.manager import AbstractPotentialManager
from GDPy.computation.driver import AbstractDriver
from GDPy.worker.worker import AbstractWorker
from GDPy.builder.builder import StructureGenerator

from GDPy.utils.command import CustomTimer

"""Monitor computation tasks with Worker.

"""

def get_file_md5(f):
    import hashlib
    m = hashlib.md5()
    while True:
        # if not using binary
        #data = f.read(1024).encode('utf-8')
        data = f.read(1024) # read in block
        if not data:
            break
        m.update(data)
    return m.hexdigest()

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

def copy_minimal_frames(prev_frames: List[Atoms]):
    """Copy atoms without extra information.

    Do not copy atoms.info since it is a dict and does not maitain order.

    """
    curr_frames, curr_info = [], []
    for prev_atoms in prev_frames:
        # - copy geometry
        curr_atoms = Atoms(
            symbols=copy.deepcopy(prev_atoms.get_chemical_symbols()),
            positions=copy.deepcopy(prev_atoms.get_positions()),
            cell=copy.deepcopy(prev_atoms.get_cell(complete=True)),
            pbc=copy.deepcopy(prev_atoms.get_pbc()),
            momenta = prev_atoms.get_momenta(), # retain this for MD
            tags = prev_atoms.get_tags() # retain this for molecules
        )
        curr_frames.append(curr_atoms)
        # - save info
        confid = prev_atoms.info.get("confid", -1)
        dynstep = prev_atoms.info.get("step", -1)
        prev_wdir = prev_atoms.info.get("wdir", "null")
        curr_info.append((confid,dynstep,prev_wdir))

    return curr_frames, curr_info


class DriverBasedWorker(AbstractWorker):

    """Monitor driver-based jobs.

    Lifetime: queued (running) -> finished -> retrieved

    Note:
        The database stores each unique job ID and its working directory.

    """

    batchsize = 1 # how many structures performed in one job

    _driver = None

    #: Whether share calc dir for each candidate. Only for command run and spc.
    _share_wdir: bool = False

    def __init__(self, potter_, driver_=None, scheduler_=None, directory_=None, *args, **kwargs):
        """"""
        self.batchsize = kwargs.pop("batchsize", 1)

        assert isinstance(potter_, AbstractPotentialManager), ""

        self.potter = potter_
        self.driver = driver_
        self.scheduler = scheduler_
        if directory_:
            self.directory = directory_
        
        self.n_jobs = config.NJOBS

        return
    
    @property
    def driver(self):
        return self._driver
    
    @driver.setter
    def driver(self, driver_):
        """"""
        assert isinstance(driver_, AbstractDriver), ""
        # TODO: check driver is consistent with potter
        self._driver = driver_
        return

    def _split_groups(self, nframes: int) -> Tuple[List[int],List[int]]:
        """Split nframes into groups."""
        # - split frames
        ngroups = int(np.floor(1.*nframes/self.batchsize))
        group_indices = [0]
        for i in range(ngroups):
            group_indices.append((i+1)*self.batchsize)
        if group_indices[-1] != nframes:
            group_indices.append(nframes)
        starts, ends = group_indices[:-1], group_indices[1:]
        assert len(starts) == len(ends), "Inconsistent start and end indices..."
        #group_indices = [f"{s}:{e}" for s, e in zip(starts,ends)]

        return (starts, ends)
    
    def _read_cached_info(self):
        # - read extra info data
        _info_data = []
        for p in (self.directory/"_data").glob("*_info.txt"):
            identifier = p.name[:self.UUIDLEN] # MD5
            with open(p, "r") as fopen:
                for line in fopen.readlines():
                    if not line.startswith("#"):
                        _info_data.append(line.strip().split())
        _info_data = sorted(_info_data, key=lambda x: int(x[0]))

        return _info_data
    
    def _preprocess(self, generator, *args, **kwargs):
        """"""
        # - get frames
        frames = []
        if isinstance(generator, StructureGenerator):
            frames = generator.run()
        else:
            assert all(isinstance(x,Atoms) for x in frames), "Input should be a list of atoms."
            frames = generator

        # - NOTE: atoms.info is a dict that does not maintain order
        #         thus, the saved file maybe different
        frames, curr_info = copy_minimal_frames(frames)

        # - check differences of input structures
        processed_dpath = self.directory/"_data"
        processed_dpath.mkdir(exist_ok=True)

        # -- get MD5 of current input structures
        # NOTE: if two jobs start too close,
        #       there may be conflicts in checking structures
        with tempfile.NamedTemporaryFile(mode="w", suffix=".xyz") as tmp:
            write(tmp.name, frames, columns=["symbols", "positions", "move_mask"])

            with open(tmp.name, "rb") as fopen:
                curr_md5 = get_file_md5(fopen)

        _info_data = self._read_cached_info()

        stored_fname = f"{curr_md5}.xyz"
        if (processed_dpath/stored_fname).exists():
            self._print(f"Found file with md5 {curr_md5}")
            self._info_data = _info_data
            start_confid = 0
            for x in self._info_data:
                if x[1] == curr_md5:
                    break
                start_confid += 1
        else:
            # - save structures
            write(
                processed_dpath/stored_fname, frames, 
                columns=["symbols", "positions", "momenta", "tags", "move_mask"]
            )
            # - save current atoms.info and append curr_info to _info_data
            start_confid = len(_info_data)
            content = "{:<12s}  {:<32s}  {:<12s}  {:<12s}  {:<s}\n".format("#id", "MD5", "confid", "step", "wdir")
            for i, (confid, step, wdir) in enumerate(curr_info):
                line = "{:<12d}  {:<32s}  {:<12d}  {:<12d}  {:<s}\n".format(i+start_confid, curr_md5, confid, step, wdir)
                content += line
                _info_data.append(line.strip().split())
            self._info_data = _info_data
            with open(processed_dpath/f"{curr_md5}_info.txt", "w") as fopen:
                fopen.write(content)

        return curr_md5, frames, start_confid
    
    def _prepare_batches(self, frames: List[Atoms], start_confid: int):
        # - check wdir
        nframes = len(frames)
        # NOTE: get a list even if it only has one structure
        # TODO: a better strategy to deal with wdirs...
        #       conflicts:
        #           merged trajectories from different drivers that all have cand0
        wdirs = [] # [(confid,dynstep), ..., ()]
        for i, atoms in enumerate(frames):
            # -- set wdir
            wdir = "cand{}".format(int(self._info_data[i+start_confid][0]))
            wdirs.append(wdir)
            atoms.info["wdir"] = wdir
        # - check whether each structure has a unique wdir
        assert len(set(wdirs)) == nframes, f"Found duplicated wdirs {len(set(wdirs))} vs. {nframes}..."

        # - split structures into different batches
        starts, ends = self._split_groups(nframes)

        batches = []
        for i, (s,e) in enumerate(zip(starts,ends)):
            # - prepare structures and dirnames
            global_indices = range(s,e)
            # NOTE: get a list even if it only has one structure
            cur_frames = [frames[x] for x in global_indices]
            cur_wdirs = [wdirs[x] for x in global_indices]
            for x in cur_frames:
                x.info["group"] = i
            # - check whether each structure has a unique wdir
            assert len(set(cur_wdirs)) == len(cur_frames), f"Found duplicated wdirs {len(set(wdirs))} vs. {len(cur_frames)} for group {i}..."

            # - set specific params
            batches.append([global_indices, cur_wdirs])

        return batches
    
    def run(self, generator=None, *args, **kwargs) -> None:
        """Split frames into groups and submit jobs.
        """
        super().run(*args, **kwargs)

        # - check if the same input structures are provided
        identifier, frames, curr_info = self._preprocess(generator)
        batches = self._prepare_batches(frames, curr_info)

        # - read metadata from file or database
        with TinyDB(
            self.directory/f"_{self.scheduler.name}_jobs.json", indent=2
        ) as database:
            queued_jobs = database.search(Query().queued.exists())
        queued_names = [q["gdir"][self.UUIDLEN+1:] for q in queued_jobs]
        queued_frames = [q["md5"] for q in queued_jobs]

        for ig, (global_indices, wdirs) in enumerate(batches):
            # - set job name
            batch_name = f"group-{ig}"
            uid = str(uuid.uuid1())
            job_name = uid + "-" + batch_name

            # -- whether store job info
            if batch_name in queued_names and identifier in queued_frames:
                self._print(f"{batch_name} at {self.directory.name} was submitted.")
                continue

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
                batch_name, uid, identifier, 
                frames, global_indices, wdirs, 
                *args, **kwargs
            )

            # - save this batch job to the database
            with TinyDB(
                self.directory/f"_{self.scheduler.name}_jobs.json", indent=2
            ) as database:
                _ = database.insert(
                    dict(
                        uid = uid,
                        md5 = identifier,
                        gdir=job_name, 
                        group_number=ig, 
                        wdir_names=wdirs, 
                        queued=True
                    )
                )

        return
    
    def _irun(self, *args, **kwargs):
        """"""

        raise NotImplementedError("Function to run a batch of structures is undefined.")
    
    def inspect(self, resubmit=False, *args, **kwargs):
        """Check if any job were finished correctly not due to time limit.

        Args:
            resubmit: Check whether submit unfinished jobs.

        """
        self._initialise(*args, **kwargs)
        self._print(f"~~~{self.__class__.__name__}+inspect")

        running_jobs = self._get_running_jobs()

        with TinyDB(
            self.directory/f"_{self.scheduler.name}_jobs.json", indent=2
        ) as database:
            for job_name in running_jobs:
                #group_directory = self.directory / job_name[self.UUIDLEN+1:]
                #group_directory = self.directory / "_works"
                group_directory = self.directory
                doc_data = database.get(Query().gdir == job_name)
                uid = doc_data["uid"]
                identifier = doc_data["md5"]

                #self.scheduler.set(**{"job-name": job_name})
                self.scheduler.job_name = job_name
                self.scheduler.script = group_directory/f"run-{uid}.script" 

                # -- check whether the jobs if running
                if self.scheduler.is_finished(): # if it is still in the queue
                    # -- valid if the task finished correctly not due to time-limit
                    is_finished = False
                    wdir_names = doc_data["wdir_names"]
                    if not self._share_wdir:
                        for x in wdir_names:
                            if not (group_directory/x).exists():
                                # even not start
                                break
                            else:
                                # not converged
                                self.driver.directory = group_directory/x
                                if not self.driver.read_convergence():
                                    break
                        else:
                            is_finished = True
                    else:
                        cache_frames = read(self.directory/"_data"/f"{identifier}_cache.xyz", ":")
                        cache_wdirs = [a.info["wdir"] for a in cache_frames]
                        if set(wdir_names) == set(cache_wdirs):
                            is_finished = True
                    if is_finished:
                        # -- finished correctly
                        self._print(f"{job_name} is finished...")
                        doc_data = database.get(Query().gdir == job_name)
                        database.update({"finished": True}, doc_ids=[doc_data.doc_id])
                    else:
                        # NOTE: no need to remove unfinished structures
                        #       since the driver would check it
                        if resubmit:
                            jobid = self.scheduler.submit()
                            self._print(f"{job_name} is re-submitted with JOBID {jobid}.")
                else:
                    self._print(f"{job_name} is running...")

        return
    
    def retrieve(self, include_retrieved: bool=False, given_wdirs: List[str]=None, *args, **kwargs) -> Trajectories:
        """Read results from wdirs.

        Args:
            include_retrieved: Whether include wdirs that are already retrieved.
                              Otherwise, all finished jobs are included.

        """
        self.inspect(*args, **kwargs)
        self._print(f"~~~{self.__class__.__name__}+retrieve")

        # NOTE: sometimes retrieve is used without run
        self._info_data = self._read_cached_info() # update _info_data

        # - check status and get latest results
        unretrieved_wdirs_ = []
        if not include_retrieved:
            unretrieved_jobs = self._get_unretrieved_jobs()
        else:
            unretrieved_jobs = self._get_finished_jobs()
            
        unretrieved_identifiers = []

        with TinyDB(
            self.directory/f"_{self.scheduler.name}_jobs.json", indent=2
        ) as database:
            for job_name in unretrieved_jobs:
                doc_data = database.get(Query().gdir == job_name)
                unretrieved_identifiers.append(doc_data["md5"])
                unretrieved_wdirs_.extend(
                    self.directory/w for w in doc_data["wdir_names"]
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
        #print("unretrieved: ", unretrieved_wdirs)
        if unretrieved_wdirs:
            unretrieved_wdirs = [pathlib.Path(x) for x in unretrieved_wdirs]
            if not self._share_wdir:
                results = self._read_results(unretrieved_wdirs, *args, **kwargs)
            else:
                # TODO: deal with traj...
                cache_frames = []
                for identifier in unretrieved_identifiers:
                    cache_frames.extend(
                        read(self.directory/"_data"/f"{identifier}_cache.xyz", ":")
                    )
                wdir_names = [x.name for x in unretrieved_wdirs]
                results = [a for a in cache_frames if a.info["wdir"] in wdir_names]
        else:
            results = []

        with TinyDB(
            self.directory/f"_{self.scheduler.name}_jobs.json", indent=2
        ) as database:
            for job_name in unretrieved_jobs:
                doc_data = database.get(Query().gdir == job_name)
                database.update({"retrieved": True}, doc_ids=[doc_data.doc_id])

        return results
    
    def _read_results(
        self, unretrieved_wdirs: List[pathlib.Path], *args, **kwargs
    ) -> Union[List[Atoms],List[List[Atoms]]]:
        """Read results from calculation directories.

        Args:
            unretrieved_wdirs: Calculation directories.

        """
        with CustomTimer(name="read-results", func=self._print):
            # NOTE: works for vasp, ...
            results_ = Parallel(n_jobs=self.n_jobs)(
                delayed(self._iread_results)(
                    self.driver, wdir, info_data = self._info_data
                ) 
                for wdir in unretrieved_wdirs
            )

            # NOTE: Failed Calcution, One fail, traj fails
            results = []
            for i, traj_frames in enumerate(results_):
                # - sift error structures
                if traj_frames:
                    error_info = traj_frames[0].info.get("error", None)
                    if error_info:
                        self._print(f"Found failed calculation at {error_info}...")
                    else:
                        results.append(traj_frames)
                else:
                    self._print(f"Found empty calculation at {str(self.directory)} with cand{i}...")

            if results:
                self._print(
                    f"new_trajectories: {len(results)} nframes of the first: {len(results[0])}"
                )
            
            # - convert to Trajectories
            results = Trajectories(results)

        return results
    
    @staticmethod
    def _iread_results(
        driver, wdir, info_data: dict = None
    ) -> List[Atoms]:
        """Extract results from a single directory.

        This must be a staticmethod as it may be pickled by joblib for parallel 
        running.

        Args:
            wdir: Working directory.

        """
        driver.directory = wdir
        # NOTE: name convention, cand1112_field1112_field1112
        confid_ = int(wdir.name.strip("cand").split("_")[0]) # internal name
        if info_data is not None: 
            cache_confid = int(info_data[confid_][2])
            if cache_confid >= 0:
                confid = cache_confid
            else:
                confid = confid_
        else:
            confid = confid_

        # NOTE: always return the entire trajectories
        traj_frames = driver.read_trajectory(add_step_info=True)
        for a in traj_frames:
            a.info["confid"] = confid
            a.info["wdir"] = str(wdir.name)
        
        return traj_frames
    
    def as_dict(self) -> dict:
        """"""
        worker_params = {}
        worker_params["potter"] = self.potter.as_dict()
        worker_params["driver"] = self.driver.as_dict()
        worker_params["scheduler"] = self.scheduler.as_dict()

        worker_params = copy.deepcopy(worker_params)

        return worker_params

class QueueDriverBasedWorker(DriverBasedWorker):

    def _irun(self, batch_name: str, uid: str, identifier: str, frames: List[Atoms], curr_indices: List[int], curr_wdirs: List[Union[str,pathlib.Path]], *args, **kwargs) -> NoReturn:
        """"""
        batch_number = int(batch_name.split("-")[-1])

        # - save worker file
        worker_params = {}
        worker_params["driver"] = self.driver.as_dict()
        worker_params["potential"] = self.potter.as_dict()
        worker_params["batchsize"] = self.batchsize

        with open(self.directory/f"worker-{uid}.yaml", "w") as fopen:
            yaml.dump(worker_params, fopen)

        # - save structures
        dataset_path = str((self.directory/"_data"/f"{identifier}.xyz").resolve())

        # - save scheduler file
        jobscript_fname = f"run-{uid}.script"
        self.scheduler.job_name = uid + "-" + batch_name
        self.scheduler.script = self.directory/jobscript_fname

        self.scheduler.user_commands = "gdp -p {} worker {} --batch {}\n".format(
            (self.directory/f"worker-{uid}.yaml").name, 
            #(self.directory/structure_fname).name
            dataset_path, batch_number
        )

        # - TODO: check whether params for scheduler is changed
        self.scheduler.write()
        if self._submit:
            self._print(f"{self.directory.name} JOBID: {self.scheduler.submit()}")
        else:
            self._print(f"{self.directory.name} waits to submit.")

        return

class CommandDriverBasedWorker(DriverBasedWorker):


    def _irun(self, batch_name: str, uid: str, identifier: str, frames: List[Atoms], curr_indices: List[int], curr_wdirs: List[Union[str,pathlib.Path]], *args, **kwargs) -> NoReturn:
        """Run calculations directly in the command line.

        Local execution supports a compact mode as structures will reuse the 
        calculation working directory.

        """
        batch_number = int(batch_name.split("-")[-1])

        # - get structures
        curr_frames = [frames[i] for i in curr_indices]

        # - run calculations
        with CustomTimer(name="run-driver", func=self._print):
            if not self._share_wdir:
                for wdir, atoms in zip(curr_wdirs,curr_frames):
                    self.driver.directory = self.directory/wdir
                    #job_name = uid+str(wdir)
                    self._print(
                        f"{time.asctime( time.localtime(time.time()) )} {str(wdir)} {self.driver.directory.name} is running..."
                    )
                    self.driver.reset()
                    self.driver.run(atoms, read_exists=True, extra_info=None)
            else:
                cache_fpath = self.directory/"_data"/f"{identifier}_cache.xyz"
                if cache_fpath.exists():
                    cache_frames = read(cache_fpath, ":")
                    cache_wdirs = [a.info["wdir"] for a in cache_frames]
                else:
                    cache_wdirs = []
                for wdir, atoms in zip(curr_wdirs,curr_frames):
                    if wdir in cache_wdirs:
                        continue
                    #if (self.directory/"_shared").exists():
                    #    shutil.rmtree(self.directory/"_shared")
                    self.driver.directory = self.directory/"_shared"
                    self._print(
                        f"{time.asctime( time.localtime(time.time()) )} {str(wdir)} {self.driver.directory.name} is running..."
                    )
                    self.driver.reset()
                    new_atoms = self.driver.run(atoms, read_exists=False, extra_info=dict(wdir=wdir))
                    # - save data
                    # TODO: There may have conflicts in write as many groups may run at the same time.
                    #       Add protection to the file.
                    write(self.directory/"_data"/f"{identifier}_cache.xyz", new_atoms, append=True)

        return

if __name__ == "__main__":
    ...