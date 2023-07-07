#!/usr/bin/env python3
# -*- coding: utf-8 -*

import pathlib
import time
from typing import Union, List
import uuid
import warnings

from tinydb import Query, TinyDB

from joblib import Parallel, delayed

from ase import Atoms

from .. import config
from ..potential.manager import AbstractPotentialManager
from .worker import AbstractWorker
from .drive import DriverBasedWorker
from ..data.array import AtomsArray2D
from ..data.array import AtomsNDArray
from ..utils.command import CustomTimer
from ..reactor.reactor import AbstractReactor


class ReactorBasedWorker(AbstractWorker):

    wdir_prefix: str = "pair" # TODO: cand?

    """Perform the computation of several reactions.
    """

    def __init__(self, potter_, driver_: AbstractReactor=None, scheduler_=None, directory_=None, *args, **kwargs):
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
    
    def _prepare_batches(self, structures: AtomsArray2D, start_confid: int):
        """"""
        nreactions = len(structures)

        wdirs = [f"{self.wdir_prefix}{i}" for i in range(nreactions)]
        
        # - split reactions into different batches
        starts, ends = self._split_groups(nreactions)

        # - 
        batches = []
        for i, (s, e) in enumerate(zip(starts,ends)):
            curr_indices = range(s, e)
            curr_wdirs = [wdirs[x] for x in curr_indices]
            batches.append([curr_indices, curr_wdirs])

        return batches
    
    def run(self, structures: List[List[Atoms]], *args, **kwargs) -> None:
        """"""
        super().run(*args, **kwargs)
        print("structures: ", structures)

        # - check if the same input structures are provided
        #identifier, frames, start_confid = self._preprocess(builder)
        identifier = "231dsa123h8931h2kjdhs78"
        batches = self._prepare_batches(structures, start_confid=0)

        # - read metadata
        with TinyDB(
            self.directory/f"_{self.scheduler.name}_jobs.json", indent=2
        ) as database:
            queued_jobs = database.search(Query().queued.exists())
        queued_names = [q["gdir"][self.UUIDLEN+1:] for q in queued_jobs]
        queued_input = [q["md5"] for q in queued_jobs]

        # -
        for i, (curr_indices, curr_wdirs) in enumerate(batches):
            # -- set job name
            batch_name = f"group-{i}"
            uid = str(uuid.uuid1())
            job_name = uid + "-" + batch_name

            # -- whether store job info
            if batch_name in queued_names and identifier in queued_input:
                self._print(f"{batch_name} at {self.directory.name} was submitted.")
                continue
                
            # -- specify which group this worker is responsible for
            #   if not, then skip
            #   Skip batch here assures the skipped batches will not recorded and 
            #   thus will not affect their execution if several batches run at the same time.
            target_number = kwargs.get("batch", None)
            if isinstance(target_number, int):
                if i != target_number:
                    with CustomTimer(name="run-driver", func=self._print):
                        self._print(
                            f"{time.asctime( time.localtime(time.time()) )} {self.driver.directory.name} batch {i} is skipped..."
                        )
                        continue
                else:
                    ...
            else:
                ...
            
            # -- run batch
            self._irun(
                batch_name, uid, identifier, structures, curr_indices, curr_wdirs,
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
                        group_number=i, 
                        wdir_names=curr_wdirs, 
                        queued=True
                    )
                )


        return
    
    def _irun(
            self, name: str, uid: str, identifier: str, structures: AtomsArray2D, 
            curr_indices, curr_wdirs, *args, **kwargs
        ) -> None:
        """"""
        with CustomTimer(name="run-reactor", func=self._print):
            # - here the driver is the reactor
            for i, wdir in zip(curr_indices, curr_wdirs):
                self._print(
                    f"{time.asctime( time.localtime(time.time()) )} {str(wdir)} {self.driver.directory.name} is running..."
                )
                self.driver.directory = self.directory/wdir
                self.driver.reset()
                _ = self.driver.run(structures[i], read_cache=True)

        return

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
                                self._print(f"{job_name} is re-submitted with JOBID {jobid}.")
                            else:
                                warnings.warn("Local scheduler does not support re-submit.", UserWarning)
                else:
                    self._print(f"{job_name} is running...")

        return
    
    def retrieve(self, include_retrieved: bool=False, given_wdirs: List[str]=None, *args, **kwargs):
        """Read results from wdirs.

        Args:
            include_retrieved: Whether include wdirs that are already retrieved.
                              Otherwise, all finished jobs are included.

        """
        self.inspect(*args, **kwargs)
        self._print(f"~~~{self.__class__.__name__}+retrieve")

        # NOTE: sometimes retrieve is used without run
        #self._info_data = self._read_cached_info() # update _info_data

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
        if unretrieved_wdirs:
            unretrieved_wdirs = [pathlib.Path(x) for x in unretrieved_wdirs]
            results = self._read_results(unretrieved_wdirs, *args, **kwargs)
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
                    self.driver, wdir, info_data = None
                ) 
                for wdir in unretrieved_wdirs
            )

            # NOTE: Failed Calcution, One fail, traj fails
            # TODO: check failed...
            #results = []
            #for i, traj_frames in enumerate(results_):
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
                self._print(
                    f"new_trajectories: {len(results)} nframes of the first: {len(results[0])}"
                )
            
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
        confid_ = int(wdir.name.strip("pair").split("_")[0]) # internal name
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
        #for a in traj_frames: # TODO: add to metadata?
        #    a.info["confid"] = confid
        #    a.info["wdir"] = str(wdir.name)
        
        return traj_frames
    
    def as_dict(self) -> dict:
        """"""
        worker_params = super().as_dict()
        worker_params["batchsize"] = self.batchsize

        return worker_params
    


if __name__ == "__main__":
    ...