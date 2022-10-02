#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import uuid
import yaml

import numpy as np

from tinydb import Query

from joblib import Parallel, delayed

from ase.io import read, write

from GDPy import config
from GDPy.potential.manager import AbstractPotentialManager
from GDPy.computation.driver import AbstractDriver
from GDPy.computation.worker.worker import AbstractWorker

from GDPy.utils.command import CustomTimer


class DriverBasedWorker(AbstractWorker):

    """ job lifetime
        queued (running) -> finished -> retrieved
    """

    batchsize = 1 # how many structures performed in one job

    _driver = None

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
                self.logger.info(f"Use attached confids...")
            else:
                wdirs = [f"cand{ia}" for ia in global_indices]
                self.logger.info(f"Use ordered confids...")

            # - prepare scheduler
            # TODO: set group name randomly?
            if self.batchsize > 1:
                group_directory = self.directory / f"g{i}" # contain one or more structures
            else: # batchsize == 1
                group_directory = self.directory / wdirs[0]
                wdirs = ["./"]
            # - set specific params
            job_info.append([group_directory, cur_frames, wdirs])
        
        # - read metadata from file or database
        queued_jobs = self.database.search(Query().queued.exists())
        queued_names = [q["gdir"][self.UUIDLEN+1:] for q in queued_jobs]

        # - run
        for group_directory, cur_frames, wdirs in job_info:
            #if job_name in self.worker_status["finished"] or job_name in self.worker_status["queued"]:
            #    continue
            if group_directory.name in queued_names:
                continue

            # - update scheduler
            job_name = str(uuid.uuid1()) + "-" + group_directory.name
            scheduler.set(**{"job-name": job_name})
            scheduler.script = group_directory/"run-driver.script" 

            # - create or check the working directory
            group_directory.mkdir()

            if scheduler.name != "local":
                cur_params = {}
                cur_params["driver"] = self.driver.as_dict()
                cur_params["potential"] = self.potter.as_dict()

                with open(group_directory/"worker.yaml", "w") as fopen:
                    yaml.dump(cur_params, fopen)

                for cur_atoms, cur_wdir in zip(cur_frames, wdirs):
                    cur_atoms.info["wdir"] = str(cur_wdir)

                write(group_directory/"frames.xyz", cur_frames)

                scheduler.user_commands = "gdp -p {} worker {}".format(
                    (group_directory/"worker.yaml").name, 
                    (group_directory/"frames.xyz").name
                )
                scheduler.write()
                if self._submit:
                    self.logger.info(f"{group_directory.name} JOBID: {scheduler.submit()}")
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

    def _read_results(
        self, gdirs, 
        read_traj=False, traj_period=1, 
        include_first=True, include_last=True
    ):
        """ wdirs - candidate dir with computation files
        """
        unretrived_wdirs = []
        for group_directory in gdirs:
            if self.batchsize > 1:
                wdirs = [x.name for x in group_directory.glob("cand*")]
            else:
                wdirs = ["./"]
            unretrived_wdirs.extend([group_directory/x for x in wdirs])

        # - get results
        results = []
        
        driver = self.driver
        with CustomTimer(name="read-results", func=self.logger.info):
            # NOTE: works for vasp, ...
            results_ = Parallel(n_jobs=self.n_jobs)(
                delayed(self._iread_results)(driver, wdir, read_traj, traj_period, include_first, include_last) 
                for wdir in unretrived_wdirs
            )

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

        return results
    
    @staticmethod
    def _iread_results(
        driver, wdir, 
        read_traj=False, traj_period=1, 
        include_first=True, include_last=True
    ):
        """"""
        driver.directory = wdir
        confid = int(wdir.name.strip("cand"))
        if not read_traj:
            new_atoms = driver.read_converged()
            new_atoms.info["confid"] = confid
            results = [new_atoms]
        else:
            traj_frames = driver.read_trajectory(add_step_info=True)
            for a in traj_frames:
                a.info["confid"] = confid
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