import copy
import dataclasses
import functools
import json
import os
import pathlib
import shlex
import shutil
import time
import traceback
import uuid
from typing import Optional, Union

import numpy as np
from ase import Atoms
from ase.io import read, write
from joblib import Parallel, delayed

from gdpx import config
from gdpx.structures.builders.builder import StructureBuilder
from gdpx.execution.driver import BaseDriver
from gdpx.execution.output import get_reporter, worker_output
from .registry import WORKER_REGISTRY
from gdpx.execution.runtime import Runtime
from gdpx.utils.archive import ZSTD_ARCHIVE_NAME, create_zstd_archive, find_driver_archive
from gdpx.utils.profiler import CustomTimer

from .store import JobRecord, JobStore
from .metadata import WorkerMetadata, CatalogJobStore
from .utils import copy_minimal_frames, split_batches
from gdpx.execution.fingerprint import (
    FINGERPRINT_VERSION, atomic_write_text, normalise_value, payload_digest, structure_digest,
    read_structure_inputs,
)
from ase.io.jsonio import decode, encode
from .worker import BaseWorker

# ---------------------------------------------------------------------------
# Module-level command-line runners
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class DriverFailure:
    """One failed structure in a driver batch."""

    computation_index: int
    workdir: str
    driver_index: int
    driver_name: str
    exception: Exception
    traceback: str


class DriverBatchError(RuntimeError):
    """Raised after every structure in a batch has been attempted."""

    def __init__(self, failures: list[DriverFailure]):
        self.failures = tuple(failures)
        details = "; ".join(
            f"{failure.workdir} ({failure.driver_name}): "
            f"{type(failure.exception).__name__}: {failure.exception}"
            for failure in self.failures
        )
        super().__init__(f"{len(self.failures)} driver computation(s) failed: {details}")


def run_computation_in_commandline(
    identifier: str,
    structures: list[Atoms],
    computation_dirnames: list[str],
    rng_states,
    drivers: Union[BaseDriver, list[BaseDriver]],
    driver_indices: Optional[list[int]],
    directory: pathlib.Path,
    share_wdir: bool,
    print_period: int = 100,
    print_func=print,
    progress_func=None,
    error_func=None,
    metadata=None,
    job_uid=None,
    machine_prefix=None,
) -> None:
    """Run computations directly in the commandline.

    Args:
        identifier: SHA-256 fingerprint of the input structures.
        structures: A batch of structures.
        computation_dirnames: Working directories for each structure.
        rng_states: Random seeds for each structure.
        drivers: A single driver or list of drivers.
        driver_indices: Which driver each structure uses (None if single driver).
        directory: Root computation directory.
        share_wdir: Whether to share a working directory.
        print_period: Print frequency for progress.
        print_func: Diagnostic print function.
        progress_func: Optional callback receiving workdir and success after each calculation.
        error_func: Optional line-by-line error sink; defaults to print_func.
    """
    is_single_driver = isinstance(drivers, BaseDriver) or len(drivers) == 1

    # Collect all unique driver instances so we can set machine_prefix once
    all_drivers: list[BaseDriver] = []
    if isinstance(drivers, BaseDriver):
        all_drivers = [drivers]
    else:
        all_drivers = drivers

    # Check machine-specific prefix
    machine_prefix_fpath = directory / "_meta" / f"MACHINE_{identifier}"
    if machine_prefix is not None:
        pass
    elif machine_prefix_fpath.exists():
        with open(machine_prefix_fpath, "r") as fopen:
            machine_prefix = "".join(fopen.readlines()).strip()
    else:
        machine_prefix = ""

    # Apply machine prefix to all drivers (save originals for restore)
    prev_prefixes = [d.setting.machine_prefix for d in all_drivers]
    if machine_prefix:
        for d in all_drivers:
            d.setting.machine_prefix = machine_prefix

    # Retrieve the right driver for a given index
    def _get_driver(idx: int) -> BaseDriver:
        if is_single_driver:
            return all_drivers[0]
        return all_drivers[idx]

    failures: list[DriverFailure] = []

    def _record_failure(gi: int, dirname: str, d_idx: int, driver: BaseDriver, error: Exception) -> None:
        traceback_text = "".join(traceback.format_exception(type(error), error, error.__traceback__))
        failure = DriverFailure(
            computation_index=gi,
            workdir=dirname,
            driver_index=d_idx,
            driver_name=getattr(driver, "name", driver.__class__.__name__),
            exception=error,
            traceback=traceback_text,
        )
        failures.append(failure)
        if error_func is None:
            print_func(f"ERROR: driver computation failed in {dirname} ({failure.driver_name})\n{traceback_text}")
        else:
            error_func(f"driver computation failed in {dirname} ({failure.driver_name})")
            for line in traceback_text.splitlines():
                error_func(line)

    # Run computations. State shared by reused driver instances is always restored.
    try:
        with CustomTimer(name="run-driver", func=print_func):
            if not share_wdir:
                for gi, (dirname, atoms, rs) in enumerate(zip(computation_dirnames, structures, rng_states)):
                    d_idx = 0 if driver_indices is None else driver_indices[gi]
                    curr_driver = _get_driver(d_idx)
                    curr_driver.directory = directory / dirname
                    succeeded = False
                    prev_random_seed = curr_driver.random_seed
                    try:
                        curr_driver.set_rng(seed=rs)
                        print_func(
                            f"{time.asctime(time.localtime(time.time()))} {dirname} "
                            f"{curr_driver.directory.name} is running..."
                        )
                        curr_driver.reset()
                        curr_driver.run(atoms, read_ckpt=True, extra_info=None)
                        succeeded = True
                    except Exception as error:
                        _record_failure(gi, dirname, d_idx, curr_driver, error)
                    finally:
                        curr_driver.set_rng(seed=prev_random_seed)
                        if progress_func is not None:
                            progress_func(dirname, succeeded)
            else:
                # shared working directory mode
                cache_fpath = directory / "_meta" / f"{identifier}_cache.xyz"
                if metadata is not None:
                    cache_wdirs = [a.info["wdir"] for a in metadata.results(job_uid)]
                elif cache_fpath.exists():
                    cache_frames = read(cache_fpath, ":")
                    cache_wdirs = [a.info["wdir"] for a in cache_frames]
                else:
                    cache_wdirs = []

                temp_wdir = directory / "_shared"
                for gi, (dirname, atoms, rs) in enumerate(zip(computation_dirnames, structures, rng_states)):
                    if dirname in cache_wdirs:
                        if progress_func is not None:
                            progress_func(dirname, True)
                        continue
                    d_idx = 0 if driver_indices is None else driver_indices[gi]
                    curr_driver = _get_driver(d_idx)
                    succeeded = False
                    prev_random_seed = curr_driver.random_seed
                    try:
                        if temp_wdir.exists():
                            shutil.rmtree(temp_wdir)
                        curr_driver.directory = temp_wdir
                        if gi % print_period == 0 or gi + 1 == len(structures):
                            print_func(
                                f"{time.asctime(time.localtime(time.time()))} {dirname} "
                                f"{curr_driver.directory.name} is running..."
                            )
                        curr_driver.set_rng(seed=rs)
                        curr_driver.reset()
                        curr_driver.run(atoms, read_ckpt=False, extra_info=dict(wdir=dirname))
                        new_atoms = curr_driver.read_trajectory()[-1]
                        new_atoms.info["wdir"] = dirname
                        if metadata is not None:
                            metadata.put_result(job_uid, new_atoms)
                        else:
                            write(cache_fpath, new_atoms, append=True)
                        succeeded = True
                    except Exception as error:
                        _record_failure(gi, dirname, d_idx, curr_driver, error)
                    finally:
                        curr_driver.set_rng(seed=prev_random_seed)
                        if progress_func is not None:
                            progress_func(dirname, succeeded)
    finally:
        for driver, previous_prefix in zip(all_drivers, prev_prefixes):
            driver.setting.machine_prefix = previous_prefix

    if failures:
        raise DriverBatchError(failures) from failures[0].exception

    return


# ---------------------------------------------------------------------------
# Unified DriverBasedWorker
# ---------------------------------------------------------------------------


@WORKER_REGISTRY.register
class DriverBasedWorker(BaseWorker):
    """Monitor driver-based jobs.

    Executes one resolved runtime over one or more input structures.

    Lifetime: queued (running) -> finished -> retrieved

    The database stores each unique job ID and its working directory.
    """

    reserved_keys: list[str] = ["energy", "step", "wdir"]
    is_spawned: bool = False
    print_period: int = 100

    _share_random_seed: bool = False
    _share_wdir: bool = False
    _retain_info: bool = False

    def __init__(
        self,
        runtime: Runtime,
        directory: Optional[Union[str, pathlib.Path]] = None,
        batchsize: int = 1,
        *args,
        **kwargs,
    ):
        super().__init__(directory=directory, batchsize=batchsize, *args, **kwargs)
        if not isinstance(runtime, Runtime):
            raise TypeError(f"Expected Runtime, got {type(runtime).__name__}.")
        self.runtime = runtime
        self.scheduler = runtime.scheduler
        self._drivers: list[BaseDriver] = [runtime.executor]

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

    def _job_reference(self, uid):
        return uid if self.compact_metadata else self.metadata_directory / f"job-{uid}.json"

    def _input_frames(self, digest):
        if self.compact_metadata:
            return self.metadata.frames(digest)
        return read_structure_inputs(self.metadata_directory / f"{digest}.atoms.json", digest)

    def _initialise(self, *args, **kwargs):
        # Do not silently start new jobs beside an old worker's records.
        legacy = list(self.directory.glob("_*_jobs.json"))
        if (self.directory / "_data").exists() or legacy:
            raise RuntimeError(
                f"Legacy driver worker layout at {self.directory}; use a new working directory."
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
        self._validate_job(job)
        self.scheduler.job_name = job.gdir
        self.scheduler.script = self._script_path(job.uid)
        self._configure_scheduler_paths()

    # ------------------------------------------------------------------
    # Driver access
    # ------------------------------------------------------------------

    @property
    def driver(self) -> BaseDriver:
        return self._drivers[0]

    @property
    def drivers(self) -> list[BaseDriver]:
        return self._drivers

    # ------------------------------------------------------------------
    # Task planning — how drivers pair with structures
    # ------------------------------------------------------------------

    def _make_task_plan(self, num_structures: int) -> list[tuple[int, int]]:
        """Build (driver_index, structure_index) pairs for each task.

        The length of the returned list is the total number of tasks.
        Each task is a single (driver, structure) computation.
        """
        return [(0, index) for index in range(num_structures)]

    # ------------------------------------------------------------------
    # Preprocessing (canonical structure caching, seed generation)
    # ------------------------------------------------------------------

    def _read_cached_info(self):
        if self.compact_metadata:
            return []
        _info_data = []
        for p in (self.metadata_directory).glob("*_info.txt"):
            with open(p, "r") as fopen:
                for line in fopen.readlines():
                    if not line.startswith("#"):
                        _info_data.append(line.strip().split())
        _info_data = sorted(_info_data, key=lambda x: int(x[0]))
        return _info_data

    def _read_cached_xinfo(self):
        info_keys, _info_data = [], []
        for p in (self.metadata_directory).glob("*_xinfo.txt"):
            with open(p, "r") as fopen:
                lines = fopen.readlines()
                info_keys = lines[0].split()[1:]
                for line in lines:
                    if not line.startswith("#"):
                        _info_data.append(line.strip().split()[1:])
        assert info_keys, "info_keys must not be empty."
        return info_keys, _info_data

    def _preprocess(self, builder, *args, **kwargs):
        frames = builder.run() if isinstance(builder, StructureBuilder) else list(builder)
        if not frames or not all(isinstance(frame, Atoms) for frame in frames):
            raise ValueError("Input should be a non-empty list of atoms.")
        prev_frames = frames

        curr_frames, curr_info = copy_minimal_frames(prev_frames)

        fingerprint = structure_digest(curr_frames)

        # Generate random seeds from the first driver's seed
        first_driver = self._drivers[0] if self._drivers else None
        if first_driver is not None:
            self._print(f"Driver's random_seed: {first_driver.random_seed}")
            if not self._share_random_seed:
                rng = np.random.Generator(np.random.PCG64(first_driver.random_seed))
                random_seeds = rng.integers(0, 1e8, size=len(curr_frames))
                random_seeds = [int(x) for x in random_seeds]
            else:
                random_seeds = [first_driver.random_seed] * len(curr_frames)
        else:
            random_seeds = [0] * len(curr_frames)

        if self.compact_metadata:
            retained = [normalise_value(a.info) for a in prev_frames] if self._retain_info else []
            self._input_provenance = curr_info
            self._input_retained = retained
            self._info_data = []
            return fingerprint, curr_frames, 0, random_seeds


    def _prepare_batches(
        self,
        frames: list[Atoms],
        start_confid: int,
        rng_states: Union[list[int], list[dict]],
    ):
        """Create batches from frames and the task plan."""
        num_frames = len(frames)
        task_plan = self._make_task_plan(num_frames)
        num_tasks = len(task_plan)

        # Build wdir names from the task plan
        wdirs = []
        driver_for_wdir = []
        struct_for_wdir = []
        for gi, (di, si) in enumerate(task_plan):
            wdir_name = "cand{}".format(gi)
            wdirs.append(wdir_name)
            driver_for_wdir.append(di)
            struct_for_wdir.append(si)
            if gi < num_frames:
                frames[si].info["wdir"] = wdir_name

        assert len(set(wdirs)) == num_tasks, f"Found duplicated wdirs {len(set(wdirs))} vs. {num_tasks}."

        overwrite_batchsize = False
        if self._share_wdir or self.scheduler.is_direct:
            overwrite_batchsize = True

        if overwrite_batchsize:
            self._print(f"Overwrites batchsize to {num_tasks=} as it uses share_wdir or direct execution.")
            batchsize_val = num_tasks
        else:
            batchsize_val = self.batchsize

        starts, ends = (
            self._split_groups(num_tasks)
            if batchsize_val == self.batchsize
            else split_batches(num_tasks, batchsize_val)
        )

        batches = []
        for i, (s, e) in enumerate(zip(starts, ends)):
            global_indices = list(range(s, e))
            batch_wdirs = [wdirs[x] for x in global_indices]
            batch_driver_indices = [driver_for_wdir[x] for x in global_indices]
            batch_seeds = []
            for gi in global_indices:
                si = struct_for_wdir[gi]
                if si < len(rng_states):
                    batch_seeds.append(rng_states[si] if isinstance(rng_states, list) else rng_states)
                else:
                    batch_seeds.append(0)
            batch_frames_structs = [
                frames[struct_for_wdir[x]] for x in global_indices if struct_for_wdir[x] < len(frames)
            ]

            for x in global_indices:
                if struct_for_wdir[x] < len(frames):
                    frames[struct_for_wdir[x]].info["group"] = i

            assert len(set(batch_wdirs)) == len(batch_wdirs), f"Found duplicated wdirs in batch {i}."

            batches.append(
                [
                    global_indices,
                    batch_wdirs,
                    batch_driver_indices,
                    batch_seeds,
                    batch_frames_structs,
                ]
            )

        # Keep task plan for later use in inspect/retrieve
        self._task_plan = task_plan

        return batches

    def prepare_batches(self, builder, rng_states=list(), *, persist=True):
        if self.metadata.root.resolve() != self.directory.resolve():
            if ((self.directory / "_meta").exists() or (self.directory / "_data").exists()
                    or any(self.directory.glob("_*_jobs.json"))):
                raise ValueError("Worker has separate metadata; use a new working directory.")
        identifier, frames, start_confid, new_rng_states = self._preprocess(builder)
        if rng_states:
            new_rng_states = rng_states
        batches = self._prepare_batches(frames, start_confid, new_rng_states)
        if persist:
            self._freeze_prepared(identifier, frames, batches, rng_states)
        return identifier, frames, batches

    def _freeze_prepared(self, identifier, frames, batches, rng_states):
        request = self._calculation_request(identifier, frames, batches, rng_states)
        saved = self.metadata.freeze_calculations([request])[self.metadata.worker]
        for batch, payload in zip(batches, saved):
            batch[3] = payload["random_seeds"]
        self._initialise()

    def _calculation_request(self, identifier, frames, batches, rng_states=()):
        return dict(worker=self.metadata.worker, frames=frames,
                    provenance=self._input_provenance, retained=self._input_retained,
                    batches=[self._job_payload(identifier, batch, index) for index, batch in enumerate(batches)],
                    machine_prefix=self.scheduler.machine_prefix,
                    reuse_saved_seeds=not rng_states and
                    self.runtime.config.executor.parameters.get("random_seed") is None)

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    @worker_output("run")
    def run(self, builder=None, rng_states=list(), *args, **kwargs) -> None:
        identifier, frames, batches = self.prepare_batches(builder, rng_states, persist=False)
        target_batch = kwargs.get("batch", None)
        if target_batch is not None and not 0 <= target_batch < len(batches):
            raise ValueError(f"Unknown batch index: {target_batch}.")
        if self.is_spawned and target_batch is None:
            raise ValueError("Spawned workers require a batch index.")
        self._freeze_prepared(identifier, frames, batches, rng_states)
        super().run(*args, **kwargs)
        selected = batches if target_batch is None else [batches[target_batch]]
        get_reporter(self).configure(selected)

        if not self.is_spawned:
            implicit_seed = self.runtime.config.executor.parameters.get("random_seed") is None
            self._run_by_scheduler(
                identifier, frames, batches, target_batch=target_batch,
                reuse_saved_seeds=not rng_states and implicit_seed,
            )
        else:
            self._run_by_commandline(identifier, frames, batches, target_batch=target_batch)

    def _run_by_commandline(self, identifier: str, frames: list[Atoms], batches, target_batch: Optional[int] = None):
        batch_data = batches[target_batch]
        compact = self.compact_metadata
        saved = None
        if compact:
            payload = self._job_payload(identifier, batch_data, target_batch)
            job_uid = self.metadata.prepare_job(payload, self.scheduler.machine_prefix)
            saved = self.metadata.manifest(job_uid)
        curr_indices, curr_wdirs, driver_indices, rng_states, curr_frames = batch_data

        run_computation_in_commandline(
            identifier,
            curr_frames,
            curr_wdirs,
            rng_states,
            self._drivers,
            driver_indices,
            self.directory,
            self._share_wdir,
            print_period=self.print_period,
            print_func=self._print,
            progress_func=get_reporter(self).task_finished,
            error_func=config.logger.error,
            metadata=self.metadata if compact else None,
            job_uid=job_uid if compact else None,
            machine_prefix=saved["machine_prefix"] if compact else None,
        )

    def _job_payload(self, identifier, batch, group_number):
        indices, wdirs, driver_indices, seeds, frames = batch
        return {
            "version": FINGERPRINT_VERSION,
            "structure_digest": identifier,
            "batch_structure_digest": structure_digest(frames),
            "runtime": normalise_value(self.as_dict()),
            "group_number": group_number,
            "indices": indices,
            "structure_indices": [self._task_plan[index][1] for index in indices],
            "wdir_names": wdirs,
            "driver_indices": driver_indices,
            "random_seeds": seeds,
            "share_random_seed": self._share_random_seed,
        }

    def _read_job_input(self, path, expected_digest=None):
        saved = self.metadata.manifest(str(path)) if self.compact_metadata else decode(pathlib.Path(path).read_text())
        payload = saved["input"]
        digest = payload_digest(payload)
        if (payload.get("version") != FINGERPRINT_VERSION or digest != saved["job_digest"]
                or (expected_digest is not None and digest != expected_digest)):
            raise ValueError(f"Job fingerprint mismatch: {path}")
        if payload_digest(payload["runtime"]) != payload_digest(self.as_dict()):
            raise ValueError("Runtime configuration changed for an existing job.")
        frames = self._input_frames(payload["structure_digest"])
        batch_frames = [frames[index].copy() for index in payload["structure_indices"]]
        if structure_digest(batch_frames) != payload["batch_structure_digest"]:
            raise ValueError("Batch structure fingerprint mismatch.")
        for frame, wdir in zip(batch_frames, payload["wdir_names"]):
            frame.info.update(wdir=wdir, group=payload["group_number"])
        batch = [payload["indices"], payload["wdir_names"], payload["driver_indices"],
                 payload["random_seeds"], batch_frames]
        return payload, batch

    def _validate_job(self, job):
        if self.compact_metadata and self.metadata.manifest(job.uid)["machine_prefix"] != self.scheduler.machine_prefix:
            raise ValueError("Machine prefix changed for an existing job.")
        payload, batch = self._read_job_input(
            self._job_reference(job.uid), job.job_digest
        )
        if (payload["structure_digest"] != job.structure_digest
                or payload["group_number"] != job.group_number
                or payload["wdir_names"] != job.wdir_names):
            raise ValueError("Job record does not match its saved input.")
        return payload, batch

    def run_saved_job(self, path):
        """Execute an immutable batch on a staged host without submitting jobs."""
        self.compact_metadata
        payload, batch = self._read_job_input(path)
        get_reporter(self).configure([batch])
        if self.compact_metadata:
            saved = self.metadata.manifest(str(path))
            run_computation_in_commandline(
                payload["structure_digest"], batch[4], batch[1], batch[3], self._drivers, batch[2],
                self.directory, self._share_wdir, print_func=self._print,
                progress_func=get_reporter(self).task_finished, error_func=config.logger.error,
                metadata=self.metadata, job_uid=str(path), machine_prefix=saved["machine_prefix"],
            )
        else:
            self._run_by_commandline(payload["structure_digest"], [], [batch], target_batch=0)

    def _run_by_scheduler(
        self, identifier: str, frames: list[Atoms], batches,
        target_batch: Optional[int] = None, *, reuse_saved_seeds: bool = False,
    ):
        # Validate the complete request before touching scheduler state or scripts.
        for index, batch in enumerate(batches):
            self.metadata.validate_payload(self._job_payload(identifier, batch, index), self.scheduler.machine_prefix)
        database_path = self.job_store.path.resolve()
        try:
            db_rel = database_path.relative_to(pathlib.Path.cwd())
        except ValueError:
            db_rel = database_path
        self._print(f"database_path: {db_rel}")

        self._input_frames(identifier)
        queued_jobs = self.job_store.get_queued()
        selected = [(ig, batch) for ig, batch in enumerate(batches)
                    if target_batch is None or ig == target_batch]
        prepared = []
        # Validate every requested batch before submitting any of them.
        for ig, batch in selected:
            payload = self._job_payload(identifier, batch, ig)
            digest = payload_digest(payload)
            overlap = [job for job in queued_jobs if set(job.wdir_names) & set(batch[1])]
            if overlap:
                if len(overlap) == 1 and reuse_saved_seeds:
                    saved, _ = self._validate_job(overlap[0])
                    # An omitted seed means resume the original random choices.
                    # All other inputs must still match exactly.
                    digest = payload_digest(dict(payload, random_seeds=saved["random_seeds"]))
                if len(overlap) != 1 or overlap[0].job_digest != digest:
                    raise ValueError(
                        f"Job input conflict for {batch[1]}: structures, runtime, or seeds changed. "
                        "Use a new working directory."
                    )
                self._validate_job(overlap[0])
                self._print(f"group-{ig} at {self.directory.name} was submitted.")
                continue
            prepared.append((ig, batch, payload, digest))

        for ig, batch, payload, digest in prepared:
            uid = (self.metadata.prepare_job(payload, self.scheduler.machine_prefix)
                   if self.compact_metadata else str(uuid.uuid1()))
            batch_name = f"group-{ig}"
            job_name = uid + "-" + batch_name
            if not self.compact_metadata:
                atomic_write_text(
                    self.metadata_directory / f"job-{uid}.json",
                    encode({"job_digest": digest, "input": payload}),
                )
            self.job_store.insert(
                uid=uid, md5="", structure_digest=identifier, job_digest=digest,
                gdir=job_name, group_number=ig, wdir_names=batch[1],
            )
            if not self.compact_metadata:
                with open(self.metadata_directory / f"MACHINE_{identifier}", "w") as handle:
                    handle.write(self.scheduler.machine_prefix)
            self._irun(batch_name, uid, identifier, frames, batch)

    def _irun(
        self,
        batch_name: str,
        uid: str,
        identifier: str,
        frames: list[Atoms],
        batch,
        *,
        submit=True,
    ) -> None:
        batch_number = int(batch_name.split("-")[-1])
        jobscript_fname = f"run-{uid}.script"
        self.scheduler.job_name = uid + "-" + batch_name
        self.scheduler.script = self._script_path(uid)
        self.scheduler.script.parent.mkdir(parents=True, exist_ok=True)
        self._configure_scheduler_paths()

        compute_plan_path = getattr(self, "compute_plan_path", None)
        if self.compact_metadata:
            self.scheduler.user_commands = f"gdp compute run --job {uid}\n"
        elif compute_plan_path is not None:
            worker_index = getattr(self, "compute_worker_index", 0)
            compute_plan_path = pathlib.Path(compute_plan_path).resolve()
            compute_root = compute_plan_path.parent.parent
            self.scheduler.local_root = compute_root
            remote_root_arg = os.path.relpath(compute_root, self.directory.resolve())
            remote_plan_arg = os.path.relpath(compute_plan_path, self.directory.resolve())
            self.scheduler.user_commands = (
                f"gdp -d {shlex.quote(remote_root_arg)} compute run "
                f"--plan {shlex.quote(remote_plan_arg)} "
                f"--worker {worker_index} --batch {batch_number}\n"
            )
        else:
            job_path = pathlib.Path("_meta") / f"job-{uid}.json"
            self.scheduler.user_commands = f"gdp compute run --job {shlex.quote(str(job_path))}\n"

        submit_variable = {"pbs": "PBS_O_WORKDIR", "slurm": "SLURM_SUBMIT_DIR",
                           "lsf": "LS_SUBCWD"}.get(self.scheduler.name)
        launch = f'cd "${{{submit_variable}:-$PWD}}" && ' if submit_variable else ""
        relative_root = "../.." if self.compact_metadata else ".."
        self.scheduler.user_commands = launch + f"cd {relative_root} && " + self.scheduler.user_commands

        curr_indices, curr_wdirs, driver_indices, rng_states, curr_frames = batch

        func_to_execute = functools.partial(
            run_computation_in_commandline,
            identifier=identifier,
            structures=curr_frames,
            computation_dirnames=curr_wdirs,
            rng_states=rng_states,
            drivers=self._drivers,
            driver_indices=driver_indices,
            directory=self.directory,
            share_wdir=self._share_wdir,
            print_period=self.print_period,
            print_func=self._print,
            progress_func=get_reporter(self).task_finished,
            error_func=config.logger.error,
            metadata=self.metadata if self.compact_metadata else None,
            job_uid=uid if self.compact_metadata else None,
            machine_prefix=(self.metadata.manifest(uid)["machine_prefix"] if self.compact_metadata else None),
        )

        self.scheduler.write()
        if not submit:
            return
        job_id = self.scheduler.submit(func_to_execute=func_to_execute)
        self.job_store.mark_submitted(self.scheduler.job_name, job_id)
        self._print(f"{self.directory.name} JOBID: {job_id}")

    # ------------------------------------------------------------------
    # Inspect
    # ------------------------------------------------------------------

    def _check_job_convergence(self, job: JobRecord) -> bool:
        wdir_names = job.wdir_names
        if not self._share_wdir:
            wdir_existence = [(self.directory / x).exists() for x in wdir_names]
            nwdir_exists = sum(1 for x in wdir_existence if x)
            self._print(f"progress: {nwdir_exists}/{len(wdir_existence)}")

            # Need the driver indices to check convergence
            # For single-driver, always use _drivers[0]
            # For multi-driver, look up from wdir name -> task plan index
            for wdir_name in wdir_names:
                if not (self.directory / wdir_name).exists():
                    return False
                # Determine driver index from wdir name
                wdir = self.directory / wdir_name
                try:
                    task_idx = int(wdir_name[4:])  # "cand{idx}"
                except ValueError:
                    return False
                if hasattr(self, "_task_plan") and task_idx < len(self._task_plan):
                    di = self._task_plan[task_idx][0]
                    d = self._drivers[di]
                else:
                    d = self._drivers[0]  # fallback for single-driver mode
                d.directory = wdir
                if not d.read_convergence():
                    self._print(f"Found unfinished computation at {wdir_name}")
                    return False
            return True
        else:
            if self.compact_metadata:
                return set(wdir_names).issubset(a.info["wdir"] for a in self.metadata.results(job.uid))
            # share_wdir: check cache file
            # Find the identifier from the job
            cache_fpath = self.metadata_directory / f"{job.structure_digest}_cache.xyz"
            if cache_fpath.exists():
                cache_frames = read(cache_fpath, ":")
                cache_wdirs = [a.info["wdir"] for a in cache_frames]
                if set(wdir_names) == set(cache_wdirs):
                    return True
                else:
                    self._print(f"Found unfinished computation at cand{len(cache_wdirs)}")
                    return False
            return False

    def _sync_job(self, job):
        self.scheduler.sync(job.wdir_names)
        if self.compact_metadata and self._share_wdir and self.scheduler.transport_name == "ssh":
            content = self.scheduler.read_remote_file("_meta/scheduler.json")
            if content is None:
                return  # The job may have failed before producing its first result.
            remote = decode(content)
            if remote.get("format") != "gdpx-scheduler" or remote.get("version") != 1:
                raise ValueError("Invalid remote result catalog.")
            self.metadata.merge_results(job.uid, remote)

    def _resubmit_job(self, job: JobRecord):
        self._print(f"RESUBMIT: {str(job.gdir)}")
        payload, batch = self._validate_job(job)
        self._irun(f"group-{job.group_number}", job.uid, payload["structure_digest"], [], batch)

    # ------------------------------------------------------------------
    # Retrieve
    # ------------------------------------------------------------------

    def _do_retrieve(
        self,
        include_retrieved: bool = False,
        given_wdirs: list[str] = None,
        use_archive: bool = False,
        *args,
        **kwargs,
    ):
        self._print(f"<<-- {self.__class__.__name__}+retrieve -->>")

        self._info_data = self._read_cached_info()

        unretrieved_wdirs_ = []
        if not include_retrieved:
            unretrieved_jobs = self.job_store.get_unretrieved()
        else:
            unretrieved_jobs = self.job_store.get_finished()

        unretrieved_identifiers = []
        for job in unretrieved_jobs:
            self._validate_job(job)
            unretrieved_identifiers.append(job.structure_digest)
            unretrieved_wdirs_.extend(self.directory / w for w in job.wdir_names)

        unretrieved_wdirs = []
        if given_wdirs is not None:
            for wdir in unretrieved_wdirs_:
                if wdir.name in given_wdirs:
                    unretrieved_wdirs.append(wdir)
        else:
            unretrieved_wdirs = unretrieved_wdirs_

        results = []
        if unretrieved_wdirs:
            unretrieved_wdirs = [pathlib.Path(x) for x in unretrieved_wdirs]
            if not self._share_wdir:
                archive_path = find_driver_archive(self.directory)
                if archive_path is None:
                    results = self._read_results(unretrieved_wdirs)
                else:
                    self._print("read archived data...")
                    results = self._read_results(unretrieved_wdirs, archive_path=archive_path)

                if use_archive and archive_path is None:
                    self._print("archive computation folders...")
                    archive_path = (self.directory / ZSTD_ARCHIVE_NAME).absolute()
                    create_zstd_archive(archive_path, ((w, w.name) for w in unretrieved_wdirs))
                    for w in unretrieved_wdirs:
                        shutil.rmtree(w)
            else:
                cache_frames = []
                if self.compact_metadata:
                    for job in unretrieved_jobs:
                        cache_frames.extend(self.metadata.results(job.uid))
                else:
                    for identifier in unretrieved_identifiers:
                        cache_frames.extend(read(self.metadata_directory / f"{identifier}_cache.xyz", ":"))
                wdir_names = [x.name for x in unretrieved_wdirs]
                results_ = [a for a in cache_frames if a.info["wdir"] in wdir_names]
                results = [[a] for a in results_]
                if self._retain_info and not self.compact_metadata:
                    info_keys, info_data = self._read_cached_xinfo()
                    retained_keys = [k for k in info_keys if k not in self.reserved_keys]
                    for i, traj_frames in enumerate(results):
                        retained_dict = {
                            k: v for k, v in zip(info_keys, info_data[i]) if k in retained_keys and v is not None
                        }
                        traj_frames[0].info.update(retained_dict)

        if self.compact_metadata:
            by_workdir = {}
            provenance = self.metadata.provenance()
            for job in unretrieved_jobs:
                saved = self.metadata.manifest(job.uid)["input"]
                info = provenance.get(job.structure_digest, {})
                for name, index in zip(saved["wdir_names"], saved["structure_indices"]):
                    by_workdir[name] = (info, index)
            for trajectory in results:
                if not trajectory:
                    continue
                info, index = by_workdir[trajectory[-1].info["wdir"]]
                rows, retained = info.get("rows", []), info.get("info", [])
                if rows and int(rows[index][0]) >= 0:
                    for frame in trajectory:
                        frame.info["confid"] = int(rows[index][0])
                if self._retain_info and retained:
                    trajectory[0].info.update({k: v for k, v in retained[index].items()
                                               if k not in self.reserved_keys})

        for job in unretrieved_jobs:
            self.job_store.mark_retrieved(job.gdir)

        return results

    def _read_results(
        self,
        unretrieved_wdirs: list[pathlib.Path],
        archive_path: pathlib.Path = None,
    ):
        with CustomTimer(name="read-results", func=self._print):
            results_ = Parallel(n_jobs=self.n_jobs)(
                delayed(self._iread_results)(
                    self._drivers,
                    wdir,
                    info_data=self._info_data,
                    archive_path=archive_path,
                )
                for wdir in unretrieved_wdirs
            )

            if self._retain_info and not self.compact_metadata:
                info_keys, info_data = self._read_cached_xinfo()
                retained_keys = [k for k in info_keys if k not in self.reserved_keys]
                for i, traj_frames in enumerate(results_):
                    retained_dict = {
                        k: v for k, v in zip(info_keys, info_data[i]) if k in retained_keys and v is not None
                    }
                    traj_frames[0].info.update(retained_dict)

            results = []
            for i, traj_frames in enumerate(results_):
                if traj_frames:
                    results.append(traj_frames)
                else:
                    self._print(f"Found empty calculation at {str(self.directory)} with cand{i}...")

            if results:
                self._print(f"new_trajectories: {len(results)} nframes of the first: {len(results[0])}")

        return results

    @staticmethod
    def _iread_results(drivers, wdir, info_data: dict = None, archive_path: pathlib.Path = None) -> list[Atoms]:
        """Extract results from a single directory.

        Must be a staticmethod — may be pickled by joblib for parallel execution.
        """
        # Determine which driver to use from the wdir name
        if isinstance(drivers, list) and len(drivers) > 1:
            try:
                task_idx = int(wdir.name[4:])
            except ValueError:
                task_idx = 0
            driver = drivers[task_idx % len(drivers)]
        else:
            driver = drivers[0] if isinstance(drivers, list) else drivers

        driver.directory = wdir
        confid_ = int(wdir.name.strip("cand").split("_")[0])
        if info_data is not None and len(info_data) > confid_:
            cache_confid = int(info_data[confid_][2]) if len(info_data[confid_]) > 2 else -1
            confid = cache_confid if cache_confid >= 0 else confid_
        else:
            confid = confid_

        traj_frames = driver.read_trajectory(add_step_info=True, archive_path=archive_path)
        for a in traj_frames:
            a.info["confid"] = confid
            a.info["wdir"] = str(wdir.name)

        return traj_frames

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def as_dict(self) -> dict:
        worker_params = self.runtime.config.to_dict()
        worker_params["options"] = {
            "batch_size": self.batchsize,
            "worker": "batch",
            "share_workdir": self._share_wdir,
            "retain_info": self._retain_info,
        }
        return copy.deepcopy(worker_params)
