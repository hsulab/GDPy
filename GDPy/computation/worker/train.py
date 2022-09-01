#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" worker for training potentials
"""

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
        queued_jobs = [q["gdir"] for q in queued_jobs]
        
        scheduler.user_commands = "\n".join(
            [self.potter.train_command, self.potter.freeze_command]
        )
        for train_dir in train_dirs:
            if train_dir.name in queued_jobs:
                continue
            # --- write files for training
            potter._make_train_files(dataset, train_dir)
            # ---
            scheduler.set(**{"job-name": train_dir.name})
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
            self.database.insert(dict(gdir=train_dir.name, queued=True))

        return
    
    def retrieve(self, *args, **kwargs):
        """"""
        super().retrieve(*args, **kwargs)
        scheduler = self.scheduler

        finished_wdirs = []
        running_jobs = self._get_running_jobs()
        for tdir_name in running_jobs:
            tdir = self.directory / tdir_name
            scheduler.set(**{"job-name": tdir_name})
            scheduler.script = tdir/"train.script" 
            if scheduler.is_finished():
                finished_wdirs.append(tdir)
                doc_data = self.database.get(Query().gdir == tdir_name)
                self.database.update({"finished": True}, doc_ids=[doc_data.doc_id])
            else:
                self.logger.info(f"{tdir_name} is running...")
        
        if finished_wdirs:
            _ = self._read_results(finished_wdirs)

        return
    
    def _read_results(self, wdirs):
        """freeze models"""
        return


if __name__ == "__main__":
    pass