import abc
import pathlib
import time
from typing import Callable, Optional, Union

import numpy as np

from gdpx import config
from gdpx.core.output import message
from gdpx.execution.output import worker_output
from gdpx.execution.schedulers import DirectScheduler
from gdpx.execution.schedulers.scheduler import BaseScheduler

from .store import JobRecord, JobStore


class BaseWorker(abc.ABC):
    """The base class of any worker using schedulers.

    Lifecycle::

        worker.run(*args, **kwargs)       → submit / run jobs
        worker.inspect(resubmit=True)     → poll for completion
        results = worker.retrieve()       → collect outputs

    Subclasses override the ``_do_run``, ``_check_job_convergence``,
    ``_resubmit_job``, and ``_do_retrieve`` hooks.
    """

    UUIDLEN = 36

    _print: Callable = config._print
    _debug: Callable = config._debug

    batchsize: int = 1

    _scheduler: BaseScheduler = DirectScheduler()
    _database = None

    _submit = True

    _script_name = "run.script"

    def __init__(
        self,
        directory: Optional[Union[str, pathlib.Path]] = None,
        batchsize=1,
    ) -> None:
        if directory is not None:
            self._directory = pathlib.Path(directory)
        else:
            self._directory = pathlib.Path.cwd() / "_tmp"

        self.batchsize = batchsize
        self.n_jobs = config.NJOBS

        self._job_store: Optional[JobStore] = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def directory(self) -> pathlib.Path:
        return self._directory

    @directory.setter
    def directory(self, directory: Union[str, pathlib.Path]):
        self._directory = pathlib.Path(directory)
        # Invalidate cached store so it picks up the new directory
        self._job_store = None

    @property
    def scheduler(self) -> BaseScheduler:
        return self._scheduler

    @scheduler.setter
    def scheduler(self, scheduler):
        assert isinstance(scheduler, BaseScheduler), ""
        self._scheduler = scheduler
        self._job_store = None

    @property
    def job_store(self) -> JobStore:
        if self._job_store is None:
            db_path = self.directory / f"_{self.scheduler.name}_jobs.json"
            self._job_store = JobStore(db_path)
        return self._job_store

    # ------------------------------------------------------------------
    # Batch helpers
    # ------------------------------------------------------------------

    def _split_groups(self, npoints: int) -> tuple[list[int], list[int]]:
        """Split npoints into groups of size *batchsize*."""
        ngroups = int(np.floor(1.0 * npoints / self.batchsize))
        group_indices = [0]
        for i in range(ngroups):
            group_indices.append((i + 1) * self.batchsize)
        if group_indices[-1] != npoints:
            group_indices.append(npoints)
        starts, ends = group_indices[:-1], group_indices[1:]

        return (starts, ends)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _initialise(self, *args, **kwargs):
        """Ensure the working directory exists."""
        if not self.directory.exists():
            self.directory.mkdir(parents=True, exist_ok=True)
        assert self.directory, "Working directory is not set properly..."

    @abc.abstractmethod
    def run(self, *args, **kwargs):
        """Submit / run all jobs.

        Subclasses must call ``super().run()`` at the start and
        implement their own submission logic.
        """
        self._initialise(*args, **kwargs)
        self._debug(f"<<-- {self.__class__.__name__}+run -->>")

    @worker_output("inspect")
    def inspect(self, resubmit=False, *args, **kwargs):
        """Check convergence of all running jobs.

        Uses the template method pattern:

        * Base class: iterates running jobs, queries the scheduler,
          marks finished in the job store.
        * Subclass hooks:
          - :meth:`_check_job_convergence` — domain-specific check.
          - :meth:`_resubmit_job` — custom resubmission.
        """
        self._initialise(*args, **kwargs)
        self._print(f"<<-- {self.__class__.__name__}+inspect -->>")

        running = self.job_store.get_running()
        remote = self.scheduler.transport_name == "ssh"
        batches = {}
        if remote and running:
            active_inputs = {job.structure_digest or job.md5 for job in running}
            queued = [job for job in self.job_store.get_queued()
                      if (job.structure_digest or job.md5) in active_inputs]
            queued.sort(key=lambda job: (job.group_number, job.uid))
            batches = {job.gdir: f"batch {index}/{len(queued)}"
                       for index, job in enumerate(queued, 1)}

        for job in running:
            self._prepare_scheduler_for_job(job)
            label = batches.get(job.gdir, job.gdir)
            finished = self.scheduler.is_finished()
            if remote:
                status = "finished" if finished else "queued/running"
                message(f"{label} | job {job.scheduler_job_id or '—'} | {status} | "
                        f"{len(job.wdir_names)} calculations")
            if finished:
                if remote:
                    message(f"{label} | fetching from {self.scheduler.hostname}...")
                    started = time.monotonic()
                self._sync_job(job)
                if remote:
                    message(f"{label} | fetched | elapsed {time.monotonic() - started:.1f} s")
                if self._check_job_convergence(job):
                    self._print(f"{job.gdir} is finished...")
                    self.job_store.mark_finished(job.gdir)
                elif resubmit:
                    if remote:
                        message(f"{label} | incomplete calculations | resubmitting...")
                    self._print(f"{job.gdir} is being re-submitted...")
                    self._resubmit_job(job)
                    if remote:
                        submitted = self.job_store.get_by_gdir(job.gdir)
                        message(f"{label} | resubmitted | job {submitted.scheduler_job_id or '—'}")
                else:
                    if remote:
                        message(f"{label} | incomplete calculations | manual resubmission required")
                    self._print(f"{job.gdir} should be re-submitted manually...")
            else:
                self._print(f"{job.gdir} is running...")

    def _sync_job(self, job):
        self.scheduler.sync(job.wdir_names)

    def _prepare_scheduler_for_job(self, job: JobRecord):
        """Set scheduler attributes from a *JobRecord*."""
        self.scheduler.job_name = job.gdir
        self.scheduler.script = self.directory / f"run-{job.uid}.script"
        compute_plan_path = getattr(self, "compute_plan_path", None)
        if compute_plan_path is not None and self.scheduler.transport_name == "ssh":
            self.scheduler.local_root = pathlib.Path(compute_plan_path).resolve().parent.parent

    def _check_job_convergence(self, job: JobRecord) -> bool:
        """Override to define domain-specific convergence logic.

        Returns:
            True if all computations in this job finished successfully.
        """
        # Default: check that all wdirs exist and the scheduler says finished
        wdir_names = job.wdir_names
        for wdir_name in wdir_names:
            if not (self.directory / wdir_name).exists():
                return False
        return True

    def _resubmit_job(self, job: JobRecord):
        """Override to define custom resubmission logic.

        The default is a no-op.
        """
        return

    @worker_output("retrieve")
    def retrieve(self, *args, **kwargs):
        """Collect results from finished jobs.

        Calls ``inspect()`` first, then delegates to :meth:`_do_retrieve`.
        """
        self.inspect(*args, **kwargs)
        self._print(f"<<-- {self.__class__.__name__}+retrieve -->>")

        return self._do_retrieve(*args, **kwargs)

    def _do_retrieve(self, *args, **kwargs):
        """Override to return results from finished jobs.

        The default returns an empty list.
        """
        return []

    # ------------------------------------------------------------------
    # Legacy job-query helpers (kept for backward compatibility)
    # ------------------------------------------------------------------

    def _get_running_jobs(self):
        return [r.gdir for r in self.job_store.get_running()]

    def _get_finished_jobs(self):
        return [r.gdir for r in self.job_store.get_finished()]

    def _get_retrieved_jobs(self):
        return [r.gdir for r in self.job_store.get_retrieved()]

    def _get_unretrieved_jobs(self):
        return [r.gdir for r in self.job_store.get_unretrieved()]

    def get_number_of_running_jobs(self):
        return len(self._get_running_jobs())

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def as_dict(self) -> dict:
        raise NotImplementedError()
