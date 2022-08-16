#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" worker = drivers + machine
"""

import pathlib
import yaml
import time

import numpy as np

from ase.io import read, write

from GDPy.utils.command import parse_input_file, CustomTimer

from GDPy.machine.factory import create_machine

from GDPy.potential.manager import PotManager

class SpecificWorker():

    """ - components
        single-frame methods
            Monte Carlo
        population methods
            GA = (generator + propagator) + driver + machine
        - machines
            moniter
            job
    """

    batchsize = 1 # how many structures performed in one job

    _directory = pathlib.Path.cwd()
    _machine = None

    prefix = "worker"
    worker_status = dict(queued=[], finished=[])

    def __init__(self, params) -> None:
        """
        """
        self.params = params

        self.prefix = params.pop("prefix", "worker")
        self.batchsize = params.pop("batchsize", 1)

        # - potter and driver
        pot_dict = params.get("potential", None).copy()
        if pot_dict is None:
            raise RuntimeError("Need potential...")
        pm = PotManager() # main potential manager
        potter = pm.create_potential(pot_name = pot_dict["name"])
        potter.register_calculator(pot_dict["params"])
        potter.version = pot_dict["version"] # NOTE: important for calculation in exp

        self.driver = potter.create_driver(params["driver"])

        # - create machine
        machine_params = params.pop("machine", None)
        self.machine = create_machine(machine_params)

        # - set default directory
        self.directory = self.directory / "MyWorker" # TODO: set dir

        return

    @property
    def directory(self):

        return self._directory
    
    @directory.setter
    def directory(self, directory_):
        """"""
        self._directory = pathlib.Path(directory_)

        return
    
    @property
    def machine(self):

        return self._machine
    
    @machine.setter
    def machine(self, machine_):
        self._machine = machine_

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
        starts, ends = self._split_groups(len(frames))

        # - create main dir
        if not self.directory.exists():
            self.directory.mkdir() # NOTE: ./tmp_folder
        else:
            pass
        
        job_info = []

        machine = self.machine
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

            # - prepare machine
            group_directory = self.directory / f"g{i}" # contain one or more structures
            # - set specific params
            job_name = self.prefix + f"-g{i}"
            job_info.append([group_directory, job_name, cur_frames, wdirs])
        
        # - read metadata from file or database
        metadata_fpath = self.directory/".metadata.yaml"
        if (metadata_fpath).exists():
            self.worker_status = parse_input_file(metadata_fpath)

        # - run or read
        for group_directory, job_name, cur_frames, wdirs in job_info:
            if job_name in self.worker_status["finished"] or job_name in self.worker_status["queued"]:
                continue
            machine.set(**{"job-name": job_name})
            machine.script = group_directory/"run-driver.script" 

            # - create or check the working directory
            group_directory.mkdir()

            if machine.name != "local":
                cur_params = self.params.copy()
                cur_params["wdirs"] = wdirs
                with open(group_directory/"driver.yaml", "w") as fopen:
                    yaml.dump(cur_params, fopen)

                write(group_directory/"frames.xyz", cur_frames)

                machine.user_commands = "gdp driver {} -s {}".format(
                    group_directory/"driver.yaml", 
                    group_directory/"frames.xyz"
                )
                machine.write()
                print(f"{group_directory.name}: ", machine.submit())

                #worker_status["queued"].extend([group_directory/x for x in wdirs])
                self.worker_status["queued"].append(job_name)
            else:
                for wdir, atoms in zip(wdirs,cur_frames):
                    #worker_status["queued"].extend([group_directory/x for x in wdirs])
                    self.driver.directory = group_directory/wdir
                    print(f"{job_name} {self.driver.directory.name} is running...")
                    self.driver.reset()
                    self.driver.run(atoms)
                self.worker_status["queued"].append(job_name)
        
        print("numebr of running jobs: ", self.get_number_of_running_jobs())

        #return

    #def retrieve(self):
        # - read metadata from file or database
        metadata_fpath = self.directory/".metadata.yaml"
        if (metadata_fpath).exists():
            self.worker_status = parse_input_file(metadata_fpath)

        # - check status and get latest results
        finished_jobnames = []
        finished_frames = []
        finished_wdirs = []
        for group_directory, job_name, cur_frames, wdirs in job_info:
            if job_name in self.worker_status["queued"] and job_name not in self.worker_status["finished"]:
                machine.set(**{"job-name": job_name})
                machine.script = group_directory/"run-driver.script" 
                if machine.is_finished():
                    print(f"{job_name} is finished...")
                    finished_wdirs.extend([group_directory/x for x in wdirs])
                    finished_frames.extend(cur_frames)
                    finished_jobnames.append(job_name)
                else:
                    print(f"{job_name} is running...")
        
        # - try to read results
        new_frames = None
        if finished_wdirs:
            new_frames = self._read_results(finished_wdirs, finished_frames)
        #print(new_frames)
        
        self.worker_status["finished"].extend(finished_jobnames)
        with open(metadata_fpath, "w") as fopen:
            yaml.dump(self.worker_status, fopen)
        
        return new_frames
    
    def _read_results(self, wdirs, frames):
        """"""
        print("ndirs: ", len(wdirs))

        results = []
        
        driver = self.driver
        with CustomTimer(name="read-results"):
            for wdir, atoms in zip(wdirs, frames):
                confid = int(wdir.name.strip("cand"))
                driver.directory = wdir
                new_atoms = driver.run(atoms, read_exsits=True)
                new_atoms.info["confid"] = confid
                print(new_atoms.info["confid"], new_atoms.get_potential_energy())
                results.append(new_atoms)

        return results
    
    def get_number_of_running_jobs(self):
        """"""
        running_jobs = [
            x for x in self.worker_status["queued"] if x not in self.worker_status["finished"]
        ]

        return len(running_jobs)

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
    worker.run(frames)

    return


if __name__ == "__main__":
    pass