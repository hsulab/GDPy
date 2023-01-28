#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
import uuid
import pathlib
import time
from typing import Tuple, List, NoReturn, Union
import yaml

import numpy as np

from tinydb import Query

from joblib import Parallel, delayed

from ase import Atoms
from ase.io import read, write

from GDPy import config
from GDPy.potential.manager import AbstractPotentialManager
from GDPy.computation.driver import AbstractDriver
from GDPy.computation.worker.worker import AbstractWorker
from GDPy.builder.builder import StructureGenerator

from GDPy.utils.command import CustomTimer

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


class DriverBasedWorker(AbstractWorker):

    """Monitor driver-based jobs.

    Lifetime: queued (running) -> finished -> retrieved

    Note:
        The database stores each unique job ID and its working directory.

    """

    batchsize = 1 # how many structures performed in one job

    _driver = None

    _exec_mode = "queue"

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
    
    def _preprocess(self, generator, *args, **kwargs):
        """"""
        # - find frames in the database

        # - get frames
        #frames, frames_fpath = [], None
        frames = []
        if isinstance(generator, StructureGenerator):
            #frames_fpath = getattr(generator, "fpath")
            frames = generator.run()
        else:
            assert all(isinstance(x,Atoms) for x in frames), "Input should be a list of atoms."
            frames = generator
        frames = copy.deepcopy(frames)
        nframes = len(frames)

        # - check wdir
        # NOTE: get a list even if it only has one structure
        wdirs = [] # [(confid,dynstep), ..., ()]
        for icand, x in enumerate(frames):
            wdir = x.info.get("wdir", None)
            if wdir is None:
                confid = x.info.get("confid", None)
                if confid:
                    dynstep = x.info.get("step", None) # step maybe 0
                    if dynstep is not None:
                        dynstep = f"_step{dynstep}"
                    else:
                        dynstep = ""
                    wdir = "cand{}{}".format(confid,dynstep)
                else:
                    wdir = f"cand{icand}"
            x.info["wdir"] = wdir
            wdirs.append(wdir)
        # - check whether each structure has a unique wdir
        assert len(set(wdirs)) == nframes, f"Found duplicated wdirs {len(set(wdirs))} vs. {nframes}..."

        # - process data
        starts, ends = self._split_groups(nframes)

        job_info = {
            "dataset": None,
            "groups": []
        }
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
            job_info["groups"].append([global_indices, cur_wdirs])

        processed_dpath = self.directory/"_data"
        processed_dpath.mkdir(exist_ok=True)

        # - check differences of input structures
        write(
            processed_dpath/"_frames.xyz", frames, columns=["symbols", "positions", "move_mask"]
        )
        with open(processed_dpath/"_frames.xyz", "rb") as fopen:
            cur_md5 = get_file_md5(fopen)
        stored_fname = f"{cur_md5}.xyz"

        job_info["dataset"] = str((processed_dpath/stored_fname).resolve())
        
        if (processed_dpath/stored_fname).exists():
            self.logger.debug(f"Found file with md5 {cur_md5}")
        else:
            write(
                processed_dpath/stored_fname, frames, columns=["symbols", "positions", "move_mask"]
            )
        # - remove temp data
        (processed_dpath/"_frames.xyz").unlink()

        return job_info
    
    def run(self, generator=None, *args, **kwargs) -> NoReturn:
        """Split frames into groups and submit jobs.
        """
        super().run(*args, **kwargs)

        # - pre
        job_info = self._preprocess(generator)

        # - check if jobs were submitted

        # - run
        if self.scheduler.name == "local":
            func = self._local_run
        else:
            func = self._queue_run
        
        _ = func(job_info)

        return 
    
    def _queue_run(self, job_info, *args, **kwargs) -> NoReturn:
        """"""
        # - read metadata from file or database
        queued_jobs = self.database.search(Query().queued.exists())
        queued_names = [q["gdir"][self.UUIDLEN+1:] for q in queued_jobs]
        queued_frames = [q["md5"] for q in queued_jobs]

        dataset = job_info["dataset"]
        cur_md5 = pathlib.Path(dataset).name.split(".")[0]
        groups = job_info["groups"]

        frames = read(dataset, ":")

        # - use shared worker parameters
        cur_params = {}
        cur_params["driver"] = self.driver.as_dict()
        cur_params["potential"] = self.potter.as_dict()
        cur_params["batchsize"] = self.batchsize

        with open(self.directory/"worker.yaml", "w") as fopen:
            yaml.dump(cur_params, fopen)

        for ig, (global_indices, wdirs) in enumerate(groups):
            # - check if already submitted
            #if job_name in self.worker_status["finished"] or job_name in self.worker_status["queued"]:
            #    continue
            group_name = f"group-{ig}"
            #if group_directory.name in queued_names:
            if group_name in queued_names and cur_md5 in queued_frames:
                self.logger.info(f"{group_name} at {self.directory.name} was submitted.")
                continue
            group_directory = self.directory

            # - update scheduler
            # NOTE: set job name
            #job_name = str(uuid.uuid1()) + "-" + group_directory.name
            uid = str(uuid.uuid1())
            job_name = uid + "-" + group_name

            # - names of input files
            jobscript_fname = f"run-{uid}.script"
            structure_fname = f"frames-{uid}.yaml"

            self.scheduler.set(**{"job-name": job_name})
            self.scheduler.script = group_directory/jobscript_fname

            # NOTE: pot file, stru file, job script
            # - use queue scheduler
            group_directory.mkdir(exist_ok=True)

            #with open(group_directory/f"g{ig}_worker.yaml", "w") as fopen:
            #    yaml.dump(cur_params, fopen)

            cur_frames = [frames[x] for x in global_indices]
            for cur_atoms, cur_wdir in zip(cur_frames, wdirs):
                cur_atoms.info["wdir"] = str(cur_wdir)

            frames_info = dict(
                method = "direct",
                frames = str(dataset),
                indices = list(global_indices)
            )
            with open(group_directory/structure_fname, "w") as fopen:
                yaml.dump(frames_info, fopen)

            self.scheduler.user_commands = "gdp -p {} driver {}\n".format(
                (group_directory/f"worker.yaml").name, 
                (group_directory/structure_fname).name
            )

            # TODO: check whether params for scheduler is changed
            self.scheduler.write()
            if self._submit:
                self.logger.info(f"{group_directory.name} JOBID: {self.scheduler.submit()}")
            else:
                self.logger.info(f"{group_directory.name} waits to submit.")

            self.database.insert(
                dict(
                    uid = uid,
                    md5 = cur_md5, # dataset md5
                    gdir=job_name, 
                    group_number=ig, 
                    wdir_names=wdirs, 
                    queued=True
                )
            )

        return
    
    def _local_run(self, job_info, *args, **kwargs) -> NoReturn:
        """"""
        dataset = job_info["dataset"]
        cur_md5 = pathlib.Path(dataset).name.split(".")[0]
        groups = job_info["groups"]

        frames = read(dataset, ":")

        for ig, (global_indices, wdirs) in enumerate(groups):
            group_name = f"group-{ig}"
            uid = str(uuid.uuid1())

            # - use local scheduler
            cur_frames = [frames[x] for x in global_indices]

            #group_directory = self.directory
            with CustomTimer(name="run-driver", func=self.logger.info):
                for wdir, atoms in zip(wdirs,cur_frames):
                    self.driver.directory = self.directory/wdir
                    job_name = uid+str(wdir)
                    self.logger.info(
                        f"{time.asctime( time.localtime(time.time()) )} {str(wdir)} {self.driver.directory.name} is running..."
                    )
                    doc_id = self.database.insert(
                        dict(
                            uid = uid,
                            md5 = cur_md5,
                            gdir=job_name, 
                            group_number=ig, 
                            wdir_names=wdirs, 
                            local=True,
                            queued=True
                        )
                    )
                    self.driver.reset()
                    self.driver.run(atoms)
                
            #self.database.update({"finished": True}, doc_ids=[doc_id])

        return

    def inspect(self, resubmit=False, *args, **kwargs):
        """Check if any job were finished correctly not due to time limit.

        Args:
            resubmit: Check whether submit unfinished jobs.

        """
        self._initialise(*args, **kwargs)
        self.logger.info(f"@@@{self.__class__.__name__}+inspect")

        running_jobs = self._get_running_jobs()
        #unretrieved_jobs = self._get_unretrieved_jobs()
        for job_name in running_jobs:
            #group_directory = self.directory / job_name[self.UUIDLEN+1:]
            #group_directory = self.directory / "_works"
            group_directory = self.directory
            doc_data = self.database.get(Query().gdir == job_name)
            uid = doc_data["uid"]

            self.scheduler.set(**{"job-name": job_name})
            self.scheduler.script = group_directory/f"run-{uid}.script" 

            # -- check whether the jobs if running
            info_name = job_name[self.UUIDLEN+1:]
            if self.scheduler.is_finished(): # if it is still in the queue
                # -- valid if the task finished correctlt not due to time-limit
                is_finished = False
                wdir_names = doc_data["wdir_names"]
                for x in wdir_names:
                    if not (group_directory/x).exists():
                        break
                else:
                    # TODO: all subtasks seem to finish...
                    is_finished = True
                if is_finished:
                    # -- finished correctly
                    self.logger.info(f"{info_name} is finished...")
                    doc_data = self.database.get(Query().gdir == job_name)
                    self.database.update({"finished": True}, doc_ids=[doc_data.doc_id])
                else:
                    # NOTE: no need to remove unfinished structures
                    #       since the driver would check it
                    if resubmit:
                        self.scheduler.submit()
                        self.logger.info(f"{info_name} is re-submitted.")
            else:
                self.logger.info(f"{info_name} is running...")

        return
    
    def retrieve(self, *args, **kwargs):
        """"""
        self.inspect(*args, **kwargs)
        self.logger.info(f"@@@{self.__class__.__name__}+retrieve")

        gdirs, results = [], []

        # - check status and get latest results
        unretrieved_wdirs = []
        unretrieved_jobs = self._get_unretrieved_jobs()
        for job_name in unretrieved_jobs:
            # NOTE: sometimes prefix has number so confid may be striped
            #group_directory = self.directory / job_name[self.UUIDLEN+1:]
            #group_directory = self.directory / "_works"
            group_directory = self.directory
            doc_data = self.database.get(Query().gdir == job_name)
            unretrieved_wdirs.extend(
                group_directory/w for w in doc_data["wdir_names"]
            )

        if unretrieved_wdirs:
            unretrieved_wdirs = [pathlib.Path(x) for x in unretrieved_wdirs]
            results = self._read_results(unretrieved_wdirs, *args, **kwargs)

        for job_name in unretrieved_jobs:
            doc_data = self.database.get(Query().gdir == job_name)
            self.database.update({"retrieved": True}, doc_ids=[doc_data.doc_id])

        return results
    
    def _read_results(
        self, unretrieved_wdirs: List[pathlib.Path], 
        read_traj: bool=False, traj_period: int=1, 
        include_first: bool=True, include_last: bool=True
    ) -> Union[List[Atoms],List[List[Atoms]]]:
        """Read results from calculation directories.

        Args:
            gdirs: A group of directories.

        """
        # - get results
        results = []
        
        driver = self.driver
        with CustomTimer(name="read-results", func=self.logger.info):
            # NOTE: works for vasp, ...
            results_ = Parallel(n_jobs=self.n_jobs)(
                delayed(self._iread_results)(driver, wdir, read_traj, traj_period, include_first, include_last) 
                for wdir in unretrieved_wdirs
            )

            if not read_traj: # read spc results, each dir has one structure
                for frames in results_:
                    # - sift error structures
                    error_info = frames[0].info.get("error", None)
                    if error_info:
                        self.logger.info(f"Found failed calculation at {error_info}...")
                    else:
                        results.extend(frames)

                if results:
                    self.logger.info(
                        f"new_frames: {len(results)} energy of the first: {results[0].get_potential_energy()}"
                    )

            else:
                for traj_frames in results_:
                    # - sift error structures
                    error_info = traj_frames[0].info.get("error", None)
                    if error_info:
                        self.logger.info(f"Found failed calculation at {error_info}...")
                    else:
                        results.append(traj_frames)

                if results:
                    self.logger.info(
                        f"new_trajectories: {len(results)} nframes of the first: {len(results[0])}"
                    )

        return results
    
    @staticmethod
    def _iread_results(
        driver, wdir, 
        read_traj: bool=False, traj_period: int=1, 
        include_first: bool=True, include_last: bool=True
    ) -> List[Atoms]:
        """Extract results from a single directory.

        Args:
            wdir: Working directory.

        """
        driver.directory = wdir
        # NOTE: name convention, cand1112_field1112_field1112
        confid = int(wdir.name.strip("cand").split("_")[0])
        if not read_traj:
            new_atoms = driver.read_converged()
            new_atoms.info["confid"] = confid
            new_atoms.info["wdir"] = str(wdir.name)
            results = [new_atoms]
        else:
            traj_frames = driver.read_trajectory(add_step_info=True)
            for a in traj_frames:
                a.info["confid"] = confid
                a.info["wdir"] = str(wdir.name)
            # NOTE: remove first or last frames since they are always the same?
            n_trajframes = len(traj_frames)
            first, last = 0, n_trajframes-1
            cur_indices = list(range(0,len(traj_frames),traj_period))
            if include_last:
                if last not in cur_indices:
                    cur_indices.append(last)
            if not include_first:
                cur_indices = cur_indices[1:]
            traj_frames = [traj_frames[i] for i in cur_indices]
            results = traj_frames

        return results

if __name__ == "__main__":
    pass