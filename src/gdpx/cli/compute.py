"""Command-line interface for schema-v3 execution lifecycles."""

from __future__ import annotations

import pathlib
from typing import Optional, Union

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
):
    """Prepare or advance one explicit schema-v3 compute lifecycle."""
    action = structures[0] if structures and structures[0] in LIFECYCLE_ACTIONS else None
    plan_path = pathlib.Path(plan) if plan is not None else pathlib.Path(directory)

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
        result = prepare_compute(load_runtime_input(runtime), structures[1:], directory)
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
        compute_plan = prepare_compute(load_runtime_input(runtime), structures, directory)
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
    box.border('bottom')
    return result
