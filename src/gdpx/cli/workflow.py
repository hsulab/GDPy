"""Command-line workflow configuration and execution."""

from __future__ import annotations

import time

import yaml

from gdpx import config
from gdpx.workflow.compiler import validate_workflow, workflow_dot, workflow_plan
from gdpx.workflow.configuration import WorkflowConfigError, WorkflowSpec, load_workflow
from gdpx.workflow.session.interface import run_workflow_spec


def parse_overrides(values: list[str] | None) -> dict:
    overrides = {}
    for value in values or []:
        if "=" not in value:
            raise WorkflowConfigError(f"Workflow override must use KEY=VALUE: {value!r}")
        key, raw = value.split("=", 1)
        key = key.strip()
        if not key:
            raise WorkflowConfigError("Workflow override key cannot be empty.")
        overrides[key] = yaml.safe_load(raw)
    return overrides


def load_cli_workflow(path, profile=None, overrides=None) -> WorkflowSpec:
    spec = load_workflow(path, profile=profile, overrides=parse_overrides(overrides))
    validate_workflow(spec)
    return spec


def run_workflow(
    path,
    *,
    profile=None,
    overrides=None,
    poll_interval: float = -1.0,
    timeout: float = -1.0,
    max_polls: int = 1000,
    directory=".",
) -> bool:
    """Run a workflow once or poll until it finishes."""
    spec = load_cli_workflow(path, profile, overrides)
    start = time.time()
    if poll_interval <= 0:
        finished = run_workflow_spec(spec, directory)
    else:
        finished = False
        for index in range(max_polls):
            config._print(f"... workflow poll {index:>04d} ...")
            finished = run_workflow_spec(spec, directory)
            if finished:
                break
            if timeout > 0 and time.time() - start + poll_interval > timeout:
                config._print("workflow reached the maximum time.")
                break
            config._print(f"... workflow will sleep for {poll_interval} seconds ...")
            time.sleep(poll_interval)
        else:
            config._print("workflow reached the maximum number of polls.")
    config._print(f"workflow time: {time.time() - start:>.4f}s")
    return finished


def validate_workflow_file(path, *, profile=None, overrides=None) -> None:
    spec = load_cli_workflow(path, profile, overrides)
    config._print(f"valid workflow: {spec.source}")


def print_workflow_plan(path, *, profile=None, overrides=None) -> None:
    spec = load_cli_workflow(path, profile, overrides)
    config._print(workflow_plan(spec))


def print_workflow_graph(path, *, profile=None, overrides=None) -> None:
    spec = load_cli_workflow(path, profile, overrides)
    config._print(workflow_dot(spec))
