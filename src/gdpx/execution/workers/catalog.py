"""Shared compact layout and scheduler setup for calculation workers."""
import pathlib

from .metadata import WorkerMetadata, CatalogJobStore
from .store import JobStore, JobRecord
from .worker import BaseWorker


class CatalogWorker(BaseWorker):
    """Keep frozen inputs, scheduler state, and job scripts under ``_meta``."""

    worker_kind = "calculation"

    @property
    def metadata(self):
        root = pathlib.Path(getattr(self, "metadata_root", self.directory))
        key = str(self.directory.resolve().relative_to(root.resolve()))
        return WorkerMetadata(root, key)

    @property
    def compact_metadata(self):
        return self.metadata.compact

    @property
    def metadata_directory(self) -> pathlib.Path:
        return self.metadata.directory

    def _script_path(self, uid):
        parent = self.metadata_directory / "jobscripts" if self.compact_metadata else self.metadata_directory
        return parent / f"run-{uid}.script"

    def _initialise(self, *args, **kwargs):
        # Do not silently start new jobs beside an old worker's records.
        legacy = list(self.directory.glob("_*_jobs.json"))
        if (self.directory / "_data").exists() or legacy:
            raise RuntimeError(
                f"Legacy {self.worker_kind} worker layout at {self.directory}; use a new working directory."
            )
        self.compact_metadata  # Reject old metadata before creating or changing files.
        workers = self.metadata.inputs.read()["workers"]
        if workers and self.metadata.worker not in workers:
            raise ValueError("Calculation set conflict: workers changed. Use a new working directory.")
        super()._initialise(*args, **kwargs)
        self.metadata_directory.mkdir(parents=True, exist_ok=True)
        if self.compact_metadata:
            self.metadata.ensure()

    @property
    def job_store(self) -> JobStore:
        self._initialise()
        if self.compact_metadata:
            if self._job_store is None:
                self._job_store = CatalogJobStore(self.metadata, self.scheduler.name)
            return self._job_store

    def _configure_scheduler_paths(self):
        if self.scheduler.transport_name == "ssh":
            plan = getattr(self, "compute_plan_path", None)
            self.scheduler.local_root = (
                pathlib.Path(plan).resolve().parent.parent if plan else self.directory.resolve()
            )
            if self.compact_metadata:
                self.scheduler.local_root = self.metadata.root.resolve()
                self.scheduler.output_root = self.directory.resolve()
                self.scheduler.staging_excludes = {
                    self.metadata.state.path.resolve(),
                    (self.metadata_directory / ".metadata.lock").resolve(),
                }
                self.scheduler.sync_excludes = {
                    self.metadata.inputs.path.resolve(), self.metadata.state.path.resolve(),
                    (self.metadata_directory / ".metadata.lock").resolve(),
                }
            else:
                self.scheduler.output_root = None
                self.scheduler.sync_excludes = set()
                self.scheduler.staging_excludes = {
                    (self.metadata_directory / "_scheduler.json").resolve()
                }

    def _prepare_scheduler_for_job(self, job: JobRecord):
        self.scheduler.job_name = job.gdir
        self.scheduler.script = self._script_path(job.uid)
        self._configure_scheduler_paths()
