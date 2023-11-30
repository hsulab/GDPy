#!/usr/bin/env python3
# -*- coding: utf-8 -*

import itertools
import pathlib
import time
import tempfile
from typing import Union, List
import uuid
import warnings
import yaml

from tinydb import Query, TinyDB

from joblib import Parallel, delayed

from ase import Atoms
from ase.io import read, write

from .. import config
from ..potential.manager import AbstractPotentialManager
from .worker import AbstractWorker
from .drive import DriverBasedWorker
from ..data.array import AtomsNDArray
from ..utils.command import CustomTimer
from ..reactor.reactor import AbstractReactor
from .utils import copy_minimal_frames, get_file_md5, read_cache_info


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
    
    def _preprocess(self, structures: Union[List[Atoms], AtomsNDArray]):
        """"""
        # TODO: For now, this only support double-ended methods, 
        #       which means the number of input structures should be even.
        if isinstance(structures, list):
            nstructures = len(structures)
            assert nstructures%2 == 0, "The number of structures should be even."
            pairs = list(
                zip(
                    [structures[i] for i in range(0, nstructures, 2)], 
                    [structures[i] for i in range(1, nstructures, 2)]
                )
            )
        elif isinstance(structures, AtomsNDArray):
            if structures.ndim == 3: # from extract
                assert structures.shape[0] == 2, "Structures must have a shape of (2, ?, ?)."
                pairs = []
                for p in structures:
                    p = [[a for a in s if a is not None][-1] for s in p]
                    pairs.append(p)
                pairs = list(zip(pairs[0], pairs[1]))
            elif structures.ndim == 2: # from extract
                pairs = list(zip(structures[0], structures[1]))
                #raise RuntimeError()
            else:
                pairs = []
                raise RuntimeError()
        else:
            ...

        # - check difference
        processed_dpath = self.directory/"_data"
        processed_dpath.mkdir(exist_ok=True)

        curr_frames, curr_info = copy_minimal_frames(itertools.chain(*pairs))

        # NOTE: handle atoms.info as some codes need energies for IS and FS...
        energies = []
        for i, a in enumerate(itertools.chain(*pairs)):
            try:
                ene = a.get_potential_energy()
                curr_frames[i].info["energy"] = ene
            except:
                ...

        with tempfile.NamedTemporaryFile(mode="w", suffix=".xyz") as tmp:
            write(tmp.name, curr_frames, columns=["symbols", "positions", "move_mask"])

            with open(tmp.name, "rb") as fopen:
                curr_md5 = get_file_md5(fopen)

        _info_data = read_cache_info(self.directory, self.UUIDLEN)
        
        cache_fname = f"{curr_md5}.xyz"
        if (processed_dpath/cache_fname).exists():
            self._print(f"Found file with md5 {curr_md5}")
            self._info_data = _info_data
            start_confid = 0
            for x in self._info_data:
                if x[1] == curr_md5:
                    break
                start_confid += 1
        else:
            write(
                processed_dpath/cache_fname, curr_frames, 
                # columns=["symbols", "positions", "momenta", "tags", "move_mask"]
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

        return curr_md5, pairs, start_confid

    def _prepare_batches(self, pairs: List[List[Atoms]], start_confid: int):
        """"""
        nreactions = len(pairs)

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

        # - check if the same input structures are provided
        identifier, pairs, start_pairid = self._preprocess(structures)
        batches = self._prepare_batches(pairs, start_confid=start_pairid)

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
            if self.scheduler.name != "local":
                if batch_name in queued_names and identifier in queued_input:
                    self._print(f"{batch_name} at {self.directory.name} was submitted.")
                    continue
            else:
                # NOTE: If use local scheduler, always run it again if re-submit
                ...
                
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
                batch_name, uid, identifier, pairs, curr_indices, curr_wdirs,
                *args, **kwargs
            )

            # - save this batch job to the database
            if identifier not in queued_input:
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
            self, name: str, uid: str, identifier: str, structures, 
            curr_indices, curr_wdirs, *args, **kwargs
        ) -> None:
        """"""
        batch_number = int(name.split("-")[-1])
        if self.scheduler.name == "local":
            with CustomTimer(name="run-reactor", func=self._print):
                # - here the driver is the reactor
                for i, wdir in zip(curr_indices, curr_wdirs):
                    self._print(
                        f"{time.asctime( time.localtime(time.time()) )} {str(wdir)} {self.driver.directory.name} is running..."
                    )
                    self.driver.directory = self.directory/wdir
                    self.driver.reset()
                    _ = self.driver.run(structures[i], read_cache=True)
        else:
            # - save worker file
            worker_params = {}
            worker_params["type"] = "reactor"
            worker_params["driver"] = self.driver.as_dict()
            worker_params["potter"] = self.potter.as_dict()
            worker_params["batchsize"] = self.batchsize

            with open(self.directory/f"worker-{uid}.yaml", "w") as fopen:
                yaml.dump(worker_params, fopen)

            # - save structures
            dataset_path = str((self.directory/"_data"/f"{identifier}.xyz").resolve())

            # - save scheduler file
            jobscript_fname = f"run-{uid}.script"
            self.scheduler.job_name = uid + "-" + name
            self.scheduler.script = self.directory/jobscript_fname

            self.scheduler.user_commands = "gdp -p {} compute {} --batch {}\n".format(
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
                ...

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
                batch = doc_data["group_number"]

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
                                #warnings.warn("Local scheduler does not support re-submit.", UserWarning)
                                frames = read(self.directory/"_data"/f"{identifier}.xyz", ":")
                                self.run(frames, batch=batch)
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
