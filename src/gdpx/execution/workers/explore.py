#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import functools
import json
import os
import pathlib
import shlex
import time
import uuid
from typing import Optional, Union

from gdpx.exploration.exploration import BaseExploration
from gdpx.exploration.layout import exploration_layout
from gdpx.execution.schedulers.scheduler import BaseScheduler

from .registry import WORKER_REGISTRY

from .worker import BaseWorker
from .store import JobStore

"""Worker that manages explorations.

Since an exploration is made up of several basic workers, this worker is a monitor that
tracks its progress.

"""


def run_exploration_in_commandline(wdir, exploration, timewait: Optional[float] = None, print_func=print) -> None:
    """"""
    exploration.directory = wdir
    if timewait is not None:
        for _ in range(1000):
            exploration.run()
            if exploration.read_convergence():
                break
            print_func(f"wait {timewait} seconds...")
            time.sleep(timewait)
        else:
            ...
    else:
        exploration.run()

    return


def get_exploration_function(
    exploration, working_directory: pathlib.Path, timewait: Optional[float] = None, print_func=print
):
    """"""
    exp_func = functools.partial(
        run_exploration_in_commandline,
        wdir=working_directory,
        exploration=exploration,
        timewait=timewait,
        print_func=print_func,
    )

    return exp_func


def save_exploration_input_parameters(inp_fpath: pathlib.Path, exploration):
    """"""
    exp_params = exploration.as_dict()
    with open(inp_fpath, "w") as fopen:
        json.dump(exp_params, fopen, indent=2)

    return


@WORKER_REGISTRY.register
class ExplorationBasedWorker(BaseWorker):

    batchsize: int = 1

    _script_name: str = "run.script"

    def __init__(
        self,
        exploration: Union[BaseExploration, list[BaseExploration]],
        scheduler: BaseScheduler,
        batchsize: int = 1,
        timewait: Optional[float] = 60.0,
        directory: Optional[Union[str, pathlib.Path]] = None,
    ) -> None:
        """"""
        super().__init__(directory=directory)

        self.exploration = exploration
        self.scheduler = scheduler

        self.batchsize = batchsize
        self.timewait = timewait

        if self.batchsize != 1:
            raise Exception("Currently, exploration worker only supports batchsize of 1.")

        return

    @property
    def explorations(self):
        return self.exploration if isinstance(self.exploration, list) else [self.exploration]

    @property
    def metadata_directory(self):
        return self.directory / '_meta'

    @property
    def job_store(self):
        self._initialise()
        if self._job_store is None:
            self._job_store = JobStore(self.metadata_directory / '_scheduler.json')
        return self._job_store

    def _initialise(self, *args, **kwargs):
        self.output_directories = exploration_layout(
            self.directory, len(self.explorations), create=True, scheduler=self.scheduler.name
        )
        super()._initialise(*args, **kwargs)

    def _prepare_job(self, index, uid, job_name, wdir_names):
        if not 0 <= index < len(self.output_directories):
            raise ValueError(f'Job {job_name} has an invalid exploration index: {index}.')
        expected = self.output_directories[index]
        if wdir_names != [expected]:
            raise ValueError(f'Job {job_name} has an inconsistent exploration directory.')
        wdir = (self.directory / expected).resolve()
        self.scheduler.job_name = job_name
        self.scheduler.script = self.metadata_directory / f'{self._script_name}-{uid}'
        if self.scheduler.transport_name == 'ssh':
            self.scheduler.local_root = self.directory.resolve()
            self.scheduler.staging_excludes = {
                path.resolve() for path in self.metadata_directory.glob('_*.json')
            }
        # Submission happens beside the script, including after SSH staging.
        # PBS may start in the user's home directory instead of that directory.
        relative_wdir = os.path.relpath(wdir, self.metadata_directory.resolve())
        input_path = os.path.relpath(self.metadata_directory.resolve() / f'exp-{uid}.json', wdir)
        wait = f' --wait {self.timewait}' if self.timewait is not None else ''
        submit_variable = {'pbs': 'PBS_O_WORKDIR', 'slurm': 'SLURM_SUBMIT_DIR',
                           'lsf': 'LS_SUBCWD'}.get(self.scheduler.name)
        launch = f'cd "${{{submit_variable}:-$PWD}}" && ' if submit_variable else ''
        self.scheduler.user_commands = (
            f'{launch}cd {shlex.quote(relative_wdir)} && '
            f'gdp explore {shlex.quote(input_path)}{wait} --spawn {index}\n'
        )
        return wdir

    def run(self, *args, **kwargs) -> None:
        super().run(*args, **kwargs)
        queued = {job.group_number: job for job in self.job_store.get_queued()}
        for index, exploration in enumerate(self.explorations):
            if index in queued:
                self._debug(f'{queued[index].gdir} at {self.directory.name} was submitted.')
                continue
            uid = str(uuid.uuid1())
            job_name = f'{uid}-expo-{index}'
            names = [self.output_directories[index]]
            wdir = self._prepare_job(index, uid, job_name, names)
            wdir.mkdir(parents=True, exist_ok=True)
            save_exploration_input_parameters(self.metadata_directory / f'exp-{uid}.json', exploration)
            # Persist scripts for direct runs too, so each job remains reproducible.
            self.scheduler.write()
            callback = get_exploration_function(exploration, wdir, self.timewait, self._print)
            status = self.scheduler.submit(func_to_execute=callback)
            self._debug(f'{names[0]}: {status}')
            self.job_store.insert(uid, '', job_name, index, names)
            self.job_store.mark_submitted(job_name, status)

    def inspect(self, resubmit=False, *args, **kwargs):
        self._initialise(*args, **kwargs)
        self._debug(f'<<-- {self.__class__.__name__}+inspect -->>')
        for job in self.job_store.get_running():
            index = job.group_number
            wdir = self._prepare_job(index, job.uid, job.gdir, job.wdir_names)
            if not self.scheduler.is_finished():
                self._print(f'{job.gdir} is running...')
                continue
            if self.scheduler.transport_name == 'ssh':
                self.scheduler.sync(job.wdir_names, root_relative=True)
            else:
                self.scheduler.sync(job.wdir_names)
            exploration = self.explorations[index]
            exploration.directory = wdir
            self._debug(f'exp_index={index}')
            self._debug(f'progress: {int(wdir.exists())}/1')
            if wdir.exists() and exploration.read_convergence():
                self.job_store.mark_finished(job.gdir)
            elif resubmit:
                wdir.mkdir(parents=True, exist_ok=True)
                self.scheduler.write()
                callback = get_exploration_function(exploration, wdir, self.timewait, self._print)
                job_id = self.scheduler.submit(func_to_execute=callback)
                self.job_store.mark_submitted(job.gdir, job_id)
                self._print(f'{job.gdir} is re-submitted with JOBID: {job_id}...')
            else:
                self._print('Resubmit is disabled.')

    def retrieve(self, include_retrieved=False, *args, **kwargs):
        self.inspect(*args, **kwargs)
        jobs = (self.job_store.get_finished() if include_retrieved
                else self.job_store.get_unretrieved())
        workers = []
        for job in jobs:
            wdir = self._prepare_job(job.group_number, job.uid, job.gdir, job.wdir_names)
            exploration = self.explorations[job.group_number]
            exploration.directory = wdir
            workers.extend(exploration.get_workers())
            self.job_store.mark_retrieved(job.gdir)
        return workers
