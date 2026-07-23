"""Persisted lifecycle for ordinary driver-based computations.

This module deliberately sits above :mod:`gdpx.worker`.  A compute plan captures
the expensive-to-reproduce choices (input structures, pairing, batches and random
seeds), while workers remain responsible for executing and reading calculations.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import pathlib
import shlex
import tempfile
import time
from typing import Iterable, Optional, Union

from ase import Atoms
from ase.io import read, write

from gdpx.factory.builder import canonicalise_builder
from gdpx.nodes.computer import ComputerVariable
from gdpx.worker.drive import DriverBasedWorker

PLAN_SCHEMA_VERSION = 1
DEFAULT_PLAN_RELPATH = pathlib.Path("_data") / "compute-plan.json"


class ComputeLifecycleError(RuntimeError):
    """Base error raised by the compute lifecycle."""


class PlanConflictError(ComputeLifecycleError):
    """A working directory already contains a different immutable plan."""


@dataclasses.dataclass(frozen=True)
class ComputeTask:
    index: int
    driver_index: int
    structure_index: int
    workdir: str
    random_seed: Union[int, dict]


@dataclasses.dataclass(frozen=True)
class ComputeBatch:
    index: int
    tasks: tuple[ComputeTask, ...]


@dataclasses.dataclass(frozen=True)
class WorkerPlan:
    index: int
    directory: str
    identifier: str
    batches: tuple[ComputeBatch, ...]


@dataclasses.dataclass(frozen=True)
class ComputePlan:
    schema_version: int
    plan_id: str
    created_at: float
    directory: str
    structure_file: str
    structure_digest: str
    config: Union[dict, list]
    workers: tuple[WorkerPlan, ...]

    @property
    def path(self) -> pathlib.Path:
        return pathlib.Path(self.directory) / DEFAULT_PLAN_RELPATH

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ComputePlan":
        workers = []
        for worker_data in data["workers"]:
            batches = []
            for batch_data in worker_data["batches"]:
                tasks = tuple(ComputeTask(**task) for task in batch_data["tasks"])
                batches.append(ComputeBatch(index=batch_data["index"], tasks=tasks))
            workers.append(
                WorkerPlan(
                    index=worker_data["index"],
                    directory=worker_data["directory"],
                    identifier=worker_data["identifier"],
                    batches=tuple(batches),
                )
            )
        return cls(
            schema_version=data["schema_version"],
            plan_id=data["plan_id"],
            created_at=data["created_at"],
            directory=data["directory"],
            structure_file=data["structure_file"],
            structure_digest=data["structure_digest"],
            config=data["config"],
            workers=tuple(workers),
        )


@dataclasses.dataclass(frozen=True)
class SubmissionResult:
    plan_id: str
    submitted_batches: tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class BatchResult:
    plan_id: str
    worker: int
    batch: int
    finished: bool


@dataclasses.dataclass(frozen=True)
class ComputeStatus:
    plan_id: str
    state: str
    queued: int
    finished: int
    retrieved: int
    total: int


@dataclasses.dataclass(frozen=True)
class ComputeResult:
    plan_id: str
    end_frames: str
    number_of_trajectories: int


def _normalise_config(config: Union[str, pathlib.Path, dict, list]) -> Union[dict, list]:
    if isinstance(config, (str, pathlib.Path)):
        from gdpx.utils.parser import parse_input_file

        parsed = parse_input_file(config)
    else:
        parsed = config
    if not isinstance(parsed, (dict, list)):
        raise ComputeLifecycleError(f"Compute configuration must be a mapping or list, got {type(parsed).__name__}.")
    normalised = copy.deepcopy(parsed)
    configs = normalised if isinstance(normalised, list) else [normalised]
    for item in configs:
        if not isinstance(item, dict):
            raise ComputeLifecycleError("Every compute configuration must be a mapping.")
        if "potter" not in item and "potential" in item:
            item["potter"] = item.pop("potential")
        if "potter" not in item:
            raise ComputeLifecycleError("Compute configuration has no `potter` (or legacy `potential`) section.")
    # Plans are JSON artifacts. Convert pathlib and scalar-like configuration
    # values once here so an in-memory plan and a reloaded plan compare equally.
    return json.loads(json.dumps(normalised, default=str))


def _load_structures(structures: Iterable[Union[str, pathlib.Path, Atoms]]) -> list[Atoms]:
    items = list(structures)
    if not items:
        raise ComputeLifecycleError("At least one input structure is required.")
    if all(isinstance(item, Atoms) for item in items):
        return [item.copy() for item in items]  # type: ignore[union-attr]

    frames: list[Atoms] = []
    for index, item in enumerate(items):
        builder = canonicalise_builder(str(item) if isinstance(item, pathlib.Path) else item)
        if builder is None:
            raise ComputeLifecycleError(f"Cannot create structures from input {item!r}.")
        builder.directory = pathlib.Path("init") / f"s{index}"
        frames.extend(builder.run())
    return frames


def _structure_digest(frames: list[Atoms]) -> str:
    with tempfile.NamedTemporaryFile(suffix=".xyz") as handle:
        write(handle.name, frames, columns=["symbols", "positions", "move_mask"])
        return hashlib.sha256(pathlib.Path(handle.name).read_bytes()).hexdigest()


def _plan_digest(payload: dict) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(canonical).hexdigest()


def _create_computer(config: Union[dict, list]):
    # Kept lazy to avoid a module cycle with the legacy compatibility adapter.
    from gdpx.cli.compute import convert_input_to_computer

    computer = convert_input_to_computer(copy.deepcopy(config))
    if not isinstance(computer, ComputerVariable):
        raise ComputeLifecycleError("The first lifecycle implementation supports ordinary computers only, not chains.")
    workers = computer.value
    if not workers or not all(isinstance(worker, DriverBasedWorker) for worker in workers):
        raise ComputeLifecycleError("The first lifecycle implementation supports DriverBasedWorker computations only.")
    return computer, workers


def _serialise_batches(worker: DriverBasedWorker, batches) -> tuple[ComputeBatch, ...]:
    serialised = []
    for batch_index, batch in enumerate(batches):
        global_indices, workdirs, driver_indices, seeds, _ = batch
        tasks = []
        for position, global_index in enumerate(global_indices):
            structure_index = int(worker._task_plan[global_index][1])
            tasks.append(
                ComputeTask(
                    index=int(global_index),
                    driver_index=int(driver_indices[position]),
                    structure_index=structure_index,
                    workdir=str(workdirs[position]),
                    random_seed=seeds[position],
                )
            )
        serialised.append(ComputeBatch(index=batch_index, tasks=tuple(tasks)))
    return tuple(serialised)


def _deserialise_batches(worker_plan: WorkerPlan, frames: list[Atoms]):
    batches = []
    for batch in worker_plan.batches:
        tasks = batch.tasks
        batches.append(
            [
                [task.index for task in tasks],
                [task.workdir for task in tasks],
                [task.driver_index for task in tasks],
                [task.random_seed for task in tasks],
                [frames[task.structure_index].copy() for task in tasks],
            ]
        )
    return batches


def _write_plan(plan: ComputePlan) -> None:
    path = plan.path
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        json.dump(plan.to_dict(), handle, indent=2)
        temp_path = pathlib.Path(handle.name)
    temp_path.replace(path)


def load_compute_plan(path_or_directory: Union[str, pathlib.Path]) -> ComputePlan:
    path = pathlib.Path(path_or_directory)
    if path.is_dir() or path.suffix != ".json":
        path = path / DEFAULT_PLAN_RELPATH
    if not path.exists():
        raise ComputeLifecycleError(f"Compute plan does not exist: {path}")
    with open(path, "r") as handle:
        data = json.load(handle)
    plan = ComputePlan.from_dict(data)
    if plan.schema_version != PLAN_SCHEMA_VERSION:
        raise ComputeLifecycleError(
            f"Unsupported compute plan schema {plan.schema_version}; expected {PLAN_SCHEMA_VERSION}."
        )
    return plan


def prepare_compute(
    config: Union[str, pathlib.Path, dict, list],
    structures: Iterable[Union[str, pathlib.Path, Atoms]],
    directory: Union[str, pathlib.Path],
) -> ComputePlan:
    directory = pathlib.Path(directory).resolve()
    normalised_config = _normalise_config(config)
    frames = _load_structures(structures)
    digest = _structure_digest(frames)

    plan_path = directory / DEFAULT_PLAN_RELPATH
    if plan_path.exists():
        existing = load_compute_plan(plan_path)
        if existing.config == normalised_config and existing.structure_digest == digest:
            return existing
        raise PlanConflictError(f"A different compute plan already exists at {plan_path}.")

    directory.mkdir(parents=True, exist_ok=True)
    input_path = directory / "_data" / "input.xyz"
    input_path.parent.mkdir(parents=True, exist_ok=True)
    write(input_path, frames)

    _, workers = _create_computer(normalised_config)
    worker_plans = []
    num_workers = len(workers)
    for worker_index, worker in enumerate(workers):
        worker_directory = directory if num_workers == 1 else directory / f"w{worker_index}"
        worker.directory = worker_directory
        worker_frames = [frame.copy() for frame in frames]
        for structure_index, frame in enumerate(worker_frames):
            frame.info["_gdpx_structure_index"] = structure_index
        identifier, _, batches = worker.prepare_batches(worker_frames)
        worker_plan = WorkerPlan(
            index=worker_index,
            directory=str(worker_directory.relative_to(directory)) if worker_directory != directory else ".",
            identifier=identifier,
            batches=_serialise_batches(worker, batches),
        )
        worker_plans.append(worker_plan)

    digest_payload = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "structure_digest": digest,
        "config": normalised_config,
        "workers": [dataclasses.asdict(worker) for worker in worker_plans],
    }
    plan = ComputePlan(
        schema_version=PLAN_SCHEMA_VERSION,
        plan_id=_plan_digest(digest_payload),
        created_at=time.time(),
        directory=str(directory),
        structure_file=str(input_path.relative_to(directory)),
        structure_digest=digest,
        config=normalised_config,
        workers=tuple(worker_plans),
    )
    _write_plan(plan)
    _write_batch_scripts(plan, workers)
    return plan


def _write_batch_scripts(plan: ComputePlan, workers: list[DriverBasedWorker]) -> None:
    """Render reviewable scripts without submitting them."""
    plan_path = plan.path
    for worker, worker_plan in zip(workers, plan.workers):
        for batch in worker_plan.batches:
            scheduler = copy.deepcopy(worker.scheduler)
            scheduler.job_name = f"gdpx-{plan.plan_id[:8]}-w{worker_plan.index}-b{batch.index}"
            command = (
                f"gdp -d {shlex.quote(plan.directory)} compute run --plan {shlex.quote(str(plan_path))} "
                f"--worker {worker_plan.index} --batch {batch.index}"
            )
            scheduler.user_commands = command + "\n"
            script_path = (
                pathlib.Path(plan.directory)
                / "_data"
                / "scripts"
                / (f"run-w{worker_plan.index}-b{batch.index}.script")
            )
            script_path.parent.mkdir(parents=True, exist_ok=True)
            content = f"#!/bin/bash -l\n\n{command}\n" if scheduler.name == "local" else str(scheduler)
            script_path.write_text(content, encoding="utf-8")


def _restore(plan: ComputePlan):
    _, workers = _create_computer(plan.config)
    if len(workers) != len(plan.workers):
        raise ComputeLifecycleError("Saved plan and reconstructed worker counts differ.")
    frames = read(pathlib.Path(plan.directory) / plan.structure_file, ":")
    restored = []
    for worker, worker_plan in zip(workers, plan.workers):
        worker.directory = pathlib.Path(plan.directory) / worker_plan.directory
        worker.compute_plan_path = plan.path
        worker.compute_worker_index = worker_plan.index
        batches = _deserialise_batches(worker_plan, frames)
        worker._task_plan = [
            (task.driver_index, task.structure_index) for batch in worker_plan.batches for task in batch.tasks
        ]
        restored.append((worker, worker_plan, batches))
    return frames, restored


def _selected_batches(worker_plan: WorkerPlan, batches: Optional[Iterable[int]]) -> list[int]:
    if batches is None:
        return [batch.index for batch in worker_plan.batches]
    selected = sorted(set(int(batch) for batch in batches))
    valid = {batch.index for batch in worker_plan.batches}
    unknown = set(selected) - valid
    if unknown:
        raise ComputeLifecycleError(f"Unknown batch indices: {sorted(unknown)}")
    return selected


def submit_compute(
    plan_or_path: Union[ComputePlan, str, pathlib.Path], batches: Optional[Iterable[int]] = None
) -> SubmissionResult:
    plan = plan_or_path if isinstance(plan_or_path, ComputePlan) else load_compute_plan(plan_or_path)
    frames, restored = _restore(plan)
    submitted = []
    for worker, worker_plan, worker_batches in restored:
        for batch_index in _selected_batches(worker_plan, batches):
            before = {record.gdir for record in worker.job_store.get_queued()}
            worker._run_by_scheduler(worker_plan.identifier, frames, worker_batches, target_batch=batch_index)
            after = {record.gdir for record in worker.job_store.get_queued()}
            if after - before:
                submitted.append(f"w{worker_plan.index}/b{batch_index}")
    return SubmissionResult(plan.plan_id, tuple(submitted))


def run_compute_batch(
    plan_or_path: Union[ComputePlan, str, pathlib.Path], batch: int, worker_index: int = 0
) -> BatchResult:
    plan = plan_or_path if isinstance(plan_or_path, ComputePlan) else load_compute_plan(plan_or_path)
    _, restored = _restore(plan)
    if worker_index < 0 or worker_index >= len(restored):
        raise ComputeLifecycleError(f"Unknown worker index: {worker_index}")
    worker, worker_plan, batches = restored[worker_index]
    _selected_batches(worker_plan, [batch])
    worker.is_spawned = True
    worker._run_by_commandline(worker_plan.identifier, [], batches, target_batch=batch)
    return BatchResult(plan.plan_id, worker_index, batch, True)


def inspect_compute(plan_or_path: Union[ComputePlan, str, pathlib.Path]) -> ComputeStatus:
    plan = plan_or_path if isinstance(plan_or_path, ComputePlan) else load_compute_plan(plan_or_path)
    _, restored = _restore(plan)
    queued = finished = retrieved = total = 0
    for worker, _, _ in restored:
        worker.inspect(resubmit=False)
        queued += len(worker.job_store.get_running())
        finished += len(worker.job_store.get_finished())
        retrieved += len(worker.job_store.get_retrieved())
        total += len(worker.job_store.get_queued())
    if total == 0:
        state = "prepared"
    elif queued:
        state = "running"
    elif finished == total:
        state = "finished"
    else:
        state = "unknown"
    return ComputeStatus(plan.plan_id, state, queued, finished, retrieved, total)


def resubmit_compute(
    plan_or_path: Union[ComputePlan, str, pathlib.Path], batches: Optional[Iterable[int]] = None
) -> SubmissionResult:
    plan = plan_or_path if isinstance(plan_or_path, ComputePlan) else load_compute_plan(plan_or_path)
    _, restored = _restore(plan)
    submitted = []
    for worker, worker_plan, _ in restored:
        selected = set(_selected_batches(worker_plan, batches))
        for job in worker.job_store.get_running():
            if job.group_number not in selected:
                continue
            worker._prepare_scheduler_for_job(job)
            if not worker.scheduler.is_finished():
                raise ComputeLifecycleError(f"Batch {job.group_number} is still running and cannot be resubmitted.")
            if worker._check_job_convergence(job):
                worker.job_store.mark_finished(job.gdir)
                continue
            worker._resubmit_job(job)
            submitted.append(f"w{worker_plan.index}/b{job.group_number}")
    return SubmissionResult(plan.plan_id, tuple(submitted))


def collect_compute(plan_or_path: Union[ComputePlan, str, pathlib.Path], archive: bool = False) -> ComputeResult:
    plan = plan_or_path if isinstance(plan_or_path, ComputePlan) else load_compute_plan(plan_or_path)
    status = inspect_compute(plan)
    if status.total == 0 or status.queued:
        raise ComputeLifecycleError(f"Cannot collect compute plan in state `{status.state}`.")

    _, restored = _restore(plan)
    trajectories = []
    for worker, _, _ in restored:
        trajectories.extend(worker.retrieve(include_retrieved=True, use_archive=archive))
    end_frames = [trajectory[-1] for trajectory in trajectories if trajectory]
    result_directory = pathlib.Path(plan.directory) / "results"
    result_directory.mkdir(parents=True, exist_ok=True)
    result_path = result_directory / "end_frames.xyz"
    write(result_path, end_frames)
    return ComputeResult(plan.plan_id, str(result_path), len(trajectories))


def orchestrate_compute(plan_or_path: Union[ComputePlan, str, pathlib.Path], archive: bool = False) -> ComputeStatus:
    """Compatibility implementation of the historical one-shot command."""
    plan = plan_or_path if isinstance(plan_or_path, ComputePlan) else load_compute_plan(plan_or_path)
    submit_compute(plan)
    _, restored = _restore(plan)
    running = 0
    for worker, _, _ in restored:
        worker.inspect(resubmit=True)
        running += worker.get_number_of_running_jobs()
    status = inspect_compute(plan)
    if status.total and running == 0:
        collect_compute(plan, archive=archive)
        status = inspect_compute(plan)
    return status
