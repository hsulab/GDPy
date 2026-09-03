"""Command-line interface for schema-v2 execution lifecycles."""

from __future__ import annotations

import dataclasses
import pathlib
from typing import Optional, Union

from gdpx import config
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
    """Prepare or advance one explicit schema-v2 compute lifecycle."""
    action = structures[0] if structures and structures[0] in LIFECYCLE_ACTIONS else None
    plan_path = pathlib.Path(plan) if plan is not None else pathlib.Path(directory)

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

    config._print(dataclasses.asdict(result))
    return result
