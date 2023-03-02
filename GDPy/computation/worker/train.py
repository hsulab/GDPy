#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" worker for training potentials
"""

import uuid

from tinydb import Query

from GDPy.computation.worker.worker import AbstractWorker
from GDPy.utils.command import run_command

class TrainWorker(AbstractWorker):

    """ components
        potter + scheduler
    """

    def __init__(self, potter_, scheduler_, directory_=None, *args, **kwargs):
        """"""
        self.potter = potter_
        self.scheduler = scheduler_
        if directory_:
            self.directory = directory_

        return
    
    def run(self, dataset=None, size=1, *args, **kwargs):
        """"""
        super().run(*args, **kwargs)

        potter = self.potter
        scheduler = self.scheduler

        train_dirs = []
        for i in range(size):
            train_dir = self.directory/("m"+str(i))
            if not train_dir.exists():
                train_dir.mkdir()
            train_dirs.append(train_dir)

        # - read metadata from file or database
        queued_jobs = self.database.search(Query().queued.exists())
        queued_names = [q["gdir"][self.UUIDLEN+1:] for q in queued_jobs]
        
        scheduler.user_commands = "\n".join(
            [self.potter.train_command, self.potter.freeze_command]
        )
        for train_dir in train_dirs:
            if train_dir.name in queued_names:
                continue
            # --- write files for training
            potter._make_train_files(dataset, train_dir)
            # ---
            job_name = str(uuid.uuid1()) + "-" + train_dir.name
            #scheduler.set(**{"job-name": job_name})
            scheduler.job_name = job_name
            scheduler.script = train_dir / "train.script"
            if scheduler.name != "local":
                scheduler.write()
                if self._submit:
                    self.logger.info(f"{train_dir.name}: {scheduler.submit()}")
                else:
                    self.logger.info(f"{train_dir.name} waits to submit.")
            else:
                # train directly
                run_command(str(train_dir), self.potter.train_command)
                run_command(str(train_dir), self.potter.freeze_command)
            self.database.insert(dict(gdir=job_name, queued=True))

        return
    
    def _read_results(self, wdirs):
        """freeze models"""
        return


if __name__ == "__main__":
    pass
