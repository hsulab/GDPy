#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" worker for running commands
"""

import uuid

from tinydb import Query

from GDPy.worker.worker import AbstractWorker
from GDPy.scheduler.scheduler import AbstractScheduler
from GDPy.utils.command import run_command

class CommandWorker(AbstractWorker):

    """ components
        potter + scheduler
    """

    def __init__(self, scheduler_: AbstractScheduler, command_: str=None, directory_=None, *args, **kwargs):
        """"""
        self.scheduler = scheduler_
        if command_:
            self._command = command_
        else:
            command_ = self.scheduler.user_commands
            if command_:
                self._command = command_
            else:
                raise RuntimeError("Cant find valid command...")
        if directory_:
            self.directory = directory_

        return
    
    @property
    def command(self):
        return self._command
    
    @command.setter
    def command(self, command_):
        self._command = command_
        return
    
    def run(self, *args, **kwargs):
        """"""
        super().run(*args, **kwargs)

        scheduler = self.scheduler

        # - read metadata from file or database
        queued_jobs = self.database.search(Query().queued.exists())
        queued_names = [q["gdir"][self.UUIDLEN+1:] for q in queued_jobs]

        wdir = self.directory
        if wdir.name in queued_names:
            pass
        else:
            job_name = str(uuid.uuid1()) + "-" + wdir.name
            scheduler.user_commands = self.command
            scheduler.set(**{"job-name": job_name})
            scheduler.script = self.directory/"run.script"
            if scheduler.name != "local":
                scheduler.write()
                if self._submit:
                    self.logger.info(f"{wdir.name}: {scheduler.submit()}")
                else:
                    self.logger.info(f"{wdir.name} waits to submit.")
            else:
                # train directly
                run_command(str(wdir), self.command)
            self.database.insert(dict(gdir=job_name, queued=True))

        return
    
    def _read_results(self, wdirs):
        """freeze models"""
        return


if __name__ == "__main__":
    pass