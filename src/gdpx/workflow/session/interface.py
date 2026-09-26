"""Execute compiled workflows with the existing session engines."""

from __future__ import annotations

import logging
import pathlib

from gdpx.workflow.compiler import CompiledWorkflow, compile_workflow
from gdpx.workflow.configuration import WorkflowSpec


def run_workflow_spec(
    spec: WorkflowSpec,
    directory: str | pathlib.Path = ".",
) -> bool:
    """Compile and run one workflow specification."""
    directory = pathlib.Path(directory)
    run_directory = directory / spec.source.stem
    settings = spec.settings
    if settings.mode == "once":
        from .once import OnceSession

        compiled: CompiledWorkflow = compile_workflow(spec, run_directory)
        session = OnceSession(directory=run_directory, stable_directories=True)
        session.run(compiled.entry, feed_dict={})
    else:
        from .repeat import RepeatSession

        session = RepeatSession(spec, directory=run_directory)
        session.run()

    # Some optional packages change logging.basicConfig and add a root handler.
    for handler in logging.root.handlers[:]:
        if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
            logging.root.removeHandler(handler)
    return session.is_finished()
