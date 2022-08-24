#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" worker = driver + scheduler
    A worker that manages a series of dynamics tasks
    worker needs a scheduler to dertermine whether run by serial
    or on cluster
"""

import pathlib
import yaml
import time

import copy

import numpy as np

from ase.io import read, write

from tinydb import TinyDB, Query

from GDPy.utils.command import parse_input_file, CustomTimer

from GDPy.scheduler.factory import create_scheduler

from GDPy.scheduler.scheduler import AbstractScheduler
from GDPy.computation.driver import AbstractDriver

from GDPy.potential.manager import PotManager


DEFAULT_MAIN_DIRNAME = "MyWorker"

class SpecificWorker():

    """ - components
        single-frame methods
            Monte Carlo
        population methods
            GA = (generator + propagator) + driver + scheduler
        - machines
            moniter
            job
    """

    batchsize = 1 # how many structures performed in one job

    _directory = None
    _scheduler = None
    _database = None

    _submit = True

    prefix = "worker"
    worker_status = dict(queued=[], finished=[])

    def __init__(self, params, directory_=None) -> None:
        """
        """
        # - pop some
        self.prefix = params.pop("prefix", "worker")
        self.batchsize = params.pop("batchsize", 1)

        # - create scheduler
        scheduler_params = params.pop("scheduler", None)
        self.scheduler = create_scheduler(scheduler_params)

        # - potter and driver
        params_ = copy.deepcopy(params)
        pot_dict = params_.get("potential", None)
        if pot_dict is None:
            raise RuntimeError("Need potential...")
        pm = PotManager() # main potential manager
        potter = pm.create_potential(pot_name = pot_dict["name"])
        potter.register_calculator(pot_dict["params"])
        potter.version = pot_dict.get("version", "unknown") # NOTE: important for calculation in exp

        self.driver = potter.create_driver(params_["driver"])

        # - set default directory
        #self.directory = self.directory / "MyWorker" # TODO: set dir
        if directory_:
            self.directory = directory_

        return

    @property
    def directory(self):

        return self._directory
    
    @directory.setter
    def directory(self, directory_):
        """"""
        # - create main dir
        directory_ = pathlib.Path(directory_)
        if not directory_.exists():
            directory_.mkdir() # NOTE: ./tmp_folder
        else:
            pass
        self._directory = directory_

        # NOTE: create a database
        self._database = TinyDB(self.directory/".metadata.json", indent=2)

        return
    
    @property
    def scheduler(self):

        return self._scheduler
    
    @scheduler.setter
    def scheduler(self, scheduelr_):
        self._scheduler = scheduelr_

        return
    
    @property
    def database(self):

        return self._database
    
    @database.setter
    def database(self, database_):
        self._database = database_

        return 
    
    def _init_database(self):
        """"""
        self.database = TinyDB(self.directory/".metadata.json", indent=2)

        return
    
    def _split_groups(self, nframes):
        """"""
        # - split frames
        ngroups = int(np.floor(1.*nframes/self.batchsize))
        group_indices = [0]
        for i in range(ngroups):
            group_indices.append((i+1)*self.batchsize)
        if group_indices[-1] != nframes:
            group_indices.append(nframes)
        #print(group_indices)
        starts, ends = group_indices[:-1], group_indices[1:]
        assert len(starts) == len(ends), "Inconsistent start and end indices..."
        #group_indices = [f"{s}:{e}" for s, e in zip(starts,ends)]
        #print(group_indices)

        return (starts, ends)
    
    def run(self, frames=None):
        """ split frames into groups
            return latest results
        """
        assert self.directory, "Working directory is not set properly..."

        scheduler = self.scheduler
        self._init_database()
        
        starts, ends = self._split_groups(len(frames))

        # - parse job info
        job_info = []
        for i, (s,e) in enumerate(zip(starts,ends)):
            # - prepare structures and dirnames
            global_indices = range(s,e)
            cur_frames = frames[s:e]
            confids = []
            for x in cur_frames:
                confid = x.info.get("confid", None)
                if confid:
                    confids.append(confid)
                else:
                    confids = []
                    break
            if confids:
                print("Use confids from structures ", confids)
                wdirs = [f"cand{ia}" for ia in confids]
            else:
                print("Use global indices ", confids)
                wdirs = [f"cand{ia}" for ia in global_indices]
            #print(list(global_indices))

            # - prepare scheduler
            # TODO: set group name randomly?
            if self.batchsize > 1:
                group_directory = self.directory / f"g{i}" # contain one or more structures
            else:
                group_directory = self.directory / wdirs[0]
                wdirs = ["./"]
            # - set specific params
            job_info.append([group_directory, cur_frames, wdirs])
        
        # - read metadata from file or database
        queued_jobs = self.database.search(Query().queued.exists())
        queued_jobs = [q["gdir"] for q in queued_jobs]
        #print("queued jobs: ", queued_jobs)

        # - run
        for group_directory, cur_frames, wdirs in job_info:
            #if job_name in self.worker_status["finished"] or job_name in self.worker_status["queued"]:
            #    continue
            job_name = self.prefix + "-" + group_directory.name
            if job_name in queued_jobs:
                continue

            # - update scheduler
            scheduler.set(**{"job-name": job_name})
            scheduler.script = group_directory/"run-driver.script" 

            # - create or check the working directory
            group_directory.mkdir()

            if scheduler.name != "local":
                cur_params = self.driver.as_dict()
                cur_params["wdirs"] = wdirs
                with open(group_directory/"driver.yaml", "w") as fopen:
                    yaml.dump(cur_params, fopen)

                write(group_directory/"frames.xyz", cur_frames)

                scheduler.user_commands = "gdp driver {} -s {}".format(
                    (group_directory/"driver.yaml").name, 
                    (group_directory/"frames.xyz").name
                )
                scheduler.write()
                if self._submit:
                    print(f"{group_directory.name}: ", scheduler.submit())
                else:
                    print(f"{group_directory.name} waits to submit.")
            else:
                for wdir, atoms in zip(wdirs,cur_frames):
                    self.driver.directory = group_directory/wdir
                    print(f"{job_name} {self.driver.directory.name} is running...")
                    self.driver.reset()
                    self.driver.run(atoms)
            self.database.insert(dict(gdir=job_name, queued=True))
        
        # - read NOTE: cant retrieve immediately
        new_frames = []
        #if self.get_number_of_running_jobs() > 0:
        #    new_frames = self.retrieve()

        return new_frames

    def retrieve(self):
        """"""
        scheduler = self.scheduler

        # - check status and get latest results
        #finished_jobnames = []
        finished_wdirs = []

        running_jobs = self._get_running_jobs()
        for job_name in running_jobs:
            # NOTE: sometimes prefix has number so confid may be striped
            group_directory = self.directory / job_name[len(self.prefix)+1:]
            #print(group_directory)
            scheduler.set(**{"job-name": job_name})
            scheduler.script = group_directory/"run-driver.script" 
            if self.batchsize > 1:
                wdirs = [x.name for x in group_directory.glob("cand*")]
            else:
                wdirs = ["./"]
            if scheduler.is_finished():
                print(f"{job_name} is finished...")
                finished_wdirs.extend([group_directory/x for x in wdirs])
                #finished_jobnames.append(job_name)
                doc_data = self.database.get(Query().gdir == job_name)
                self.database.update({"finished": True}, doc_ids=[doc_data.doc_id])
            else:
                print(f"{job_name} is running...")
        
        # - try to read results
        new_frames = []
        if finished_wdirs:
            new_frames = self._read_results(finished_wdirs)
            print("new_frames: ", len(new_frames), new_frames[0].get_potential_energy())
        #print("new_frames: ", new_frames)
        
        return new_frames
    
    def _read_results(self, wdirs):
        """ wdirs - candidate dir with computation files
        """
        #print("ndirs: ", len(wdirs))

        results = []
        
        driver = self.driver
        with CustomTimer(name="read-results"):
            for wdir in wdirs:
                print("wdir: ", wdir)
                confid = int(wdir.name.strip("cand"))
                driver.directory = wdir
                #new_atoms = driver.run(atoms, read_exsits=True)
                new_atoms = driver.read_converged()
                new_atoms.info["confid"] = confid
                #print(new_atoms.info["confid"], new_atoms.get_potential_energy())
                results.append(new_atoms)

        return results

    def _get_running_jobs(self):
        """"""
        running_jobs = self.database.search(
            Query().queued.exists() and (~Query().finished.exists())
        )
        running_jobs = [r["gdir"] for r in running_jobs]

        return running_jobs
    
    def get_number_of_running_jobs(self):
        """"""
        running_jobs = self._get_running_jobs()

        return len(running_jobs)


class DriverBasedWorker(SpecificWorker):

    def __init__(self, driver_, scheduler_, directory_=None, *args, **kwargs):
        """"""
        assert isinstance(driver_, AbstractDriver), ""
        assert isinstance(scheduler_, AbstractScheduler), ""

        self.driver = driver_
        self.scheduler = scheduler_
        if directory_:
            self.directory = directory_

        return

def create_worker(params: dict):
    """"""
    worker = SpecificWorker(params)

    return worker

def run_worker(params: str, structure: str, potter=None):
    """"""
    # - check if params are all valid
    params = parse_input_file(params)
    """ # TODO: check params
    if potter is None:
        pot_dict = params.get("potential", None)
        if pot_dict is None:
            raise RuntimeError("Need potential...")
        pm = PotManager() # main potential manager
        potter = pm.create_potential(pot_name = pot_dict["name"])
        potter.register_calculator(pot_dict["params"])
        potter.version = pot_dict["version"] # NOTE: important for calculation in exp
    """

    worker = SpecificWorker(params)

    # TODO: 
    frames = read(structure, ":")

    # - find input frames
    worker.directory = pathlib.Path.cwd() / DEFAULT_MAIN_DIRNAME

    worker.run(frames)
    if worker.get_number_of_running_jobs() > 0:
        new_frames = worker.retrieve()
        if new_frames:
            write(worker.directory/"new_frames.xyz", new_frames, append=True)

    return


if __name__ == "__main__":
    pass