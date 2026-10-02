"""Command-line interface for versioned execution lifecycles."""

from __future__ import annotations

import pathlib
import time
from typing import Optional, Union

from ase import Atoms
from ase.io import write

from gdpx.core.output import Box
from gdpx.execution.output import reporting_session
from gdpx.execution.lifecycle import (
    collect_compute,
    inspect_compute,
    orchestrate_compute,
    prepare_compute,
    resubmit_compute,
    run_compute_batch,
    submit_compute,
)
from gdpx.utils.parser import parse_input_file


DEFAULT_MAIN_DIRNAME = "MyWorker"
LIFECYCLE_ACTIONS = {"prepare", "submit", "run", "status", "resubmit", "collect"}


def load_runtime_input(value):
    """Load a runtime mapping or explicit runtime list without adapting legacy forms."""
    return parse_input_file(value) if isinstance(value, (str, pathlib.Path)) else value


def _run_reactor_once(worker, structures, directory, archive=False):
    """Run the historical reactor worker behind the one-shot compute interface."""
    from gdpx.execution.lifecycle import ComputeResult, ComputeStatus
    from gdpx.structures.builders.factory import canonicalise_builder

    frames = []
    for source in structures:
        builder = canonicalise_builder(source)
        if builder is None:
            raise RuntimeError(f"Cannot create structures from input {source!r}.")
        frames.extend(builder.run())
    if len(frames) < 2:
        raise RuntimeError("A reactor computation requires at least two path images.")

    runtime = worker.runtime.config.to_dict()
    method = runtime["executor"]["method"]
    box = Box(f"worker | {method}")
    box.line(
        f"executor: {runtime['executor']['provider']}   "
        f"potential: {runtime['potential']['provider']}"
    )
    box.line("state: running")
    started = time.monotonic()
    previous_print = worker.driver._print
    worker.driver._print = box.line
    try:
        worker.run(frames)
        worker.inspect(resubmit=True)
        running = worker.get_number_of_running_jobs()
        total = len(worker.job_store.get_queued())
        if running:
            box.line(f"state: waiting   pending: {running}")
            return ComputeStatus("reactor", "running", running, 0, 0, total)

        trajectories = worker.retrieve(include_retrieved=True, use_archive=archive)
        end_frames = []
        for trajectory in trajectories:
            if not trajectory:
                continue
            final = trajectory[-1]
            if isinstance(final, Atoms):
                end_frames.append(final)
            else:
                end_frames.extend(final)
        result_directory = pathlib.Path(directory) / "results"
        result_directory.mkdir(parents=True, exist_ok=True)
        result_path = result_directory / "end_frames.xyz"
        write(result_path, end_frames)
        box.line(f"state: finished   paths: {len(trajectories)}")
        return ComputeResult("reactor", str(result_path), len(trajectories))
    except Exception:
        box.line(f"state: failed   details: {directory}")
        raise
    finally:
        worker.driver._print = previous_print
        box.border("bottom", f"elapsed: {time.monotonic() - started:.1f} s", align="right")


@reporting_session
def run_computation(
    structures,
    runtime,
    *,
    batch: Optional[int] = None,
    spawn: bool = False,
    archive: bool = False,
    directory: Union[str, pathlib.Path] = pathlib.Path.cwd() / DEFAULT_MAIN_DIRNAME,
    plan: Optional[Union[str, pathlib.Path]] = None,
    worker_index: int = 0,
    job: Optional[Union[str, pathlib.Path]] = None,
    task: Optional[int] = None,
    random_provenance: Optional[dict] = None,
):
    """Prepare or advance one explicit compute lifecycle."""
    action = structures[0] if structures and structures[0] in LIFECYCLE_ACTIONS else None
    plan_path = pathlib.Path(plan) if plan is not None else pathlib.Path(directory)
    compute_plan = None

    if job is not None:
        if action != "run" or plan is not None:
            raise ValueError("--job requires compute run and cannot be combined with --plan.")
        from gdpx.execution.factory import create_worker
        from gdpx.execution.fingerprint import FINGERPRINT_VERSION, payload_digest

        from gdpx.execution.workers.metadata import WorkerMetadata

        metadata = WorkerMetadata(directory)
        by_uuid = pathlib.Path(str(job)).suffix != ".json"
        if by_uuid:
            saved = metadata.validate_manifest(metadata.inputs.read()["jobs"][str(job)])
            saved = WorkerMetadata(directory, saved["worker"]).manifest(str(job))
        else:
            raise ValueError("Legacy job manifests are not supported; use a new working directory.")
        if (saved["input"].get("version") != FINGERPRINT_VERSION
                or payload_digest(saved["input"]) != saved["job_digest"]):
            raise ValueError(f"Job fingerprint mismatch: {job}")
        worker_directory = pathlib.Path(directory) / saved["worker"] if by_uuid else directory
        worker = create_worker(saved["input"]["runtime"], directory=worker_directory)
        if by_uuid:
            worker.metadata_root = pathlib.Path(directory)
        worker.run_saved_job(job, task=task)
        return

    if task is not None:
        raise ValueError("--task requires --job.")

    if spawn and action is None:
        if runtime is None or batch is None:
            raise RuntimeError("Spawned computations require --runtime and --batch.")
        from gdpx.execution.factory import create_worker
        from gdpx.structures.builders.factory import canonicalise_builder

        worker = create_worker(load_runtime_input(runtime), directory=directory)
        worker.is_spawned = True
        frames = []
        for source in structures:
            frames.extend(canonicalise_builder(source).run())
        worker.run(frames, batch=batch)
        return

    if action == "prepare":
        if runtime is None:
            raise RuntimeError("`gdp compute prepare` requires `--runtime`.")
        result = prepare_compute(
            load_runtime_input(runtime), structures[1:], directory,
            random_provenance=random_provenance,
        )
    elif action == "submit":
        result = submit_compute(plan_path, batches=None if batch is None else [batch])
    elif action == "run":
        if batch is None:
            raise RuntimeError("`gdp compute run` requires `--batch`.")
        result = run_compute_batch(plan_path, batch=batch, worker_index=worker_index)
    elif action == "status":
        result = inspect_compute(plan_path)
    elif action == "resubmit":
        result = resubmit_compute(plan_path, batches=None if batch is None else [batch])
    elif action == "collect":
        result = collect_compute(plan_path, archive=archive)
    else:
        if runtime is None:
            raise RuntimeError("`gdp compute` requires `--runtime`.")
        runtime_input = load_runtime_input(runtime)
        from gdpx.execution.factory import create_worker
        from gdpx.execution.workers.react import ReactorBasedWorker

        candidate = None if isinstance(runtime_input, (list, tuple)) else create_worker(
            runtime_input, directory=directory
        )
        if isinstance(candidate, ReactorBasedWorker):
            result = _run_reactor_once(candidate, structures, directory, archive=archive)
        else:
            compute_plan = prepare_compute(
                runtime_input, structures, directory,
                random_provenance=random_provenance,
            )
            if spawn:
                result = submit_compute(compute_plan, batches=None if batch is None else [batch])
            else:
                result = orchestrate_compute(compute_plan, archive=archive)

    box = Box('compute | ' + (action or 'run'))
    if hasattr(result, 'state'):
        box.line(f'state: {result.state}   batches: {result.total}   finished: {result.finished}   pending: {result.queued}')
    elif hasattr(result, 'number_of_trajectories'):
        box.line(f'collected: {result.number_of_trajectories} calculations')
        box.line(f'results: {result.end_frames}')
    elif hasattr(result, 'submitted_batches'):
        box.line(f'submitted batches: {len(result.submitted_batches)}')
    elif hasattr(result, 'workers'):
        box.line(f'workers: {len(result.workers)}   batches: {sum(len(w.batches) for w in result.workers)}')
        box.line(f'plan: {result.path}')
    else:
        box.line(f'worker: {result.worker}   batch: {result.batch}   finished: {str(result.finished).lower()}')
    seed_plan = result if hasattr(result, 'workers') else compute_plan
    if seed_plan is not None:
        task_count = sum(len(batch.tasks) for worker in seed_plan.workers for batch in worker.batches)
        task_label = 'task' if task_count == 1 else 'tasks'
        seed_catalog = seed_plan.path.relative_to(pathlib.Path(seed_plan.directory))
        box.line(f'random seeds: {task_count} {task_label} recorded in {seed_catalog}')
    box.border('bottom')
    return result
