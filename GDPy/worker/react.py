#!/usr/bin/env python3
# -*- coding: utf-8 -*

import time
from typing import List
import uuid

from tinydb import Query, TinyDB

from ase import Atoms

from .. import config
from ..potential.manager import AbstractPotentialManager
from .worker import AbstractWorker
from .drive import DriverBasedWorker
from ..data.array import AtomsArray2D
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


if __name__ == "__main__":
    ...