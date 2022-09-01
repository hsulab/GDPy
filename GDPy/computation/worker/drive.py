#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import yaml

import numpy as np

from tinydb import Query

from ase.io import read, write

from GDPy.potential.potential import AbstractPotential
from GDPy.computation.driver import AbstractDriver
from GDPy.computation.worker.worker import AbstractWorker

from GDPy.utils.command import CustomTimer


class DriverBasedWorker(AbstractWorker):

    batchsize = 1 # how many structures performed in one job

    _driver = None

    def __init__(self, potter_, driver_=None, scheduler_=None, directory_=None, *args, **kwargs):
        """"""
        self.batchsize = kwargs.pop("batchsize", 1)

        assert isinstance(potter_, AbstractPotential), ""

        self.potter = potter_
        self.driver = driver_
        self.scheduler = scheduler_
        if directory_:
            self.directory = directory_

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

    def _split_groups(self, nframes):
        """"""
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
    
    def run(self, frames=None, *args, **kwargs):
        """ split frames into groups
            return latest results
        """
        super().run(*args, **kwargs)
        scheduler = self.scheduler
        
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
                wdirs = [f"cand{ia}" for ia in confids]
            else:
                wdirs = [f"cand{ia}" for ia in global_indices]
            self.logger.info(f"Use confids as {wdirs}")

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
                    self.logger.info(f"{group_directory.name}: {scheduler.submit()}")
                else:
                    self.logger.info(f"{group_directory.name} waits to submit.")
            else:
                for wdir, atoms in zip(wdirs,cur_frames):
                    self.driver.directory = group_directory/wdir
                    self.logger.info(f"{job_name} {self.driver.directory.name} is running...")
                    self.driver.reset()
                    self.driver.run(atoms)
            self.database.insert(dict(gdir=job_name, queued=True))
        
        return 

    def retrieve(self, *args, **kwargs):
        """"""
        super().retrieve(*args, **kwargs)

        scheduler = self.scheduler

        # - check status and get latest results
        #finished_jobnames = []
        finished_wdirs = []

        running_jobs = self._get_running_jobs()
        for job_name in running_jobs:
            # NOTE: sometimes prefix has number so confid may be striped
            group_directory = self.directory / job_name[len(self.prefix)+1:]
            scheduler.set(**{"job-name": job_name})
            scheduler.script = group_directory/"run-driver.script" 
            if self.batchsize > 1:
                wdirs = [x.name for x in group_directory.glob("cand*")]
            else:
                wdirs = ["./"]
            if scheduler.is_finished():
                self.logger.info(f"{job_name} is finished...")
                finished_wdirs.extend([group_directory/x for x in wdirs])
                #finished_jobnames.append(job_name)
                doc_data = self.database.get(Query().gdir == job_name)
                self.database.update({"finished": True}, doc_ids=[doc_data.doc_id])
            else:
                self.logger.info(f"{job_name} is running...")
        
        # - try to read results
        new_frames = []
        if finished_wdirs:
            new_frames = self._read_results(finished_wdirs)
            self.logger.info(f"new_frames: {len(new_frames)} {new_frames[0].get_potential_energy()}")
        
        return new_frames
    
    def _read_results(self, wdirs, read_traj=False):
        """ wdirs - candidate dir with computation files
        """
        # TODO: add if retrieved in metadata
        results = []
        
        driver = self.driver
        with CustomTimer(name="read-results", func=self.logger.info):
            for wdir in wdirs:
                driver.directory = wdir
                confid = int(wdir.name.strip("cand"))
                if not read_traj:
                    new_atoms = driver.read_converged()
                    new_atoms.info["confid"] = confid
                    results.append(new_atoms)
                else:
                    # TODO: remove first or last frames since they are always the same?
                    traj_frames = driver.read_trajectory(add_step_info=True) # TODO: add step to info
                    for a in traj_frames:
                        a.info["confid"] = confid
                    results.extend(traj_frames)

        return results

if __name__ == "__main__":
    pass