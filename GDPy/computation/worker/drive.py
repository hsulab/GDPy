#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import yaml

import numpy as np

from tinydb import Query

from ase.io import read, write

from GDPy.scheduler.scheduler import AbstractScheduler
from GDPy.computation.driver import AbstractDriver
from GDPy.computation.worker.worker import AbstractWorker

from GDPy.utils.command import CustomTimer


class DriverBasedWorker(AbstractWorker):

    def __init__(self, driver_, scheduler_, directory_=None, *args, **kwargs):
        """"""
        assert isinstance(driver_, AbstractDriver), ""
        assert isinstance(scheduler_, AbstractScheduler), ""

        self.driver = driver_
        self.scheduler = scheduler_
        if directory_:
            self.directory = directory_

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

if __name__ == "__main__":
    pass