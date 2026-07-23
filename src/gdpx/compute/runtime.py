"""Runtime services shared by CLI commands and domain workflows."""

from __future__ import annotations

import copy
import enum
import logging
import pathlib

from ase import Atoms
from ase.io import read, write

from gdpx.factory.builder import canonicalise_builder
from gdpx.factory.computer import create_worker_chains, create_workers
from gdpx.reactor.reactor import BaseReactor
from gdpx.utils.parser import parse_input_file


class CompState(enum.Enum):
    QUEUED = "queued"
    FINISHED = "finished"


def create_computer(config):
    """Create workers or worker chains without mutating input configuration."""
    if isinstance(config, (str, pathlib.Path)):
        config = parse_input_file(input_fpath=config)
    config = copy.deepcopy(config)
    if isinstance(config, dict):
        return create_workers(config)
    if isinstance(config, list) and config:
        return create_worker_chains(config) if len(config) > 1 else create_workers(config[0])
    raise TypeError(f"Computer configuration must be a non-empty mapping or list, got {type(config).__name__}.")


def run_one_worker(structures, worker, directory, batch=None, spawn=False, archive=False):
    directory = pathlib.Path(directory)
    worker.directory = directory
    worker.is_spawned = spawn
    worker.run(structures, batch=batch)
    state = CompState.QUEUED
    if not spawn:
        worker.inspect(resubmit=True, batch=batch)
        if worker.get_number_of_running_jobs() == 0:
            results = directory / "results"
            if not results.exists():
                results.mkdir()
                trajectories = worker.retrieve(include_retrieved=True, use_archive=archive)
                if not isinstance(worker.driver, BaseReactor):
                    write(results / "end_frames.xyz", [trajectory[-1] for trajectory in trajectories])
                    state = CompState.FINISHED
            else:
                state = CompState.FINISHED
    return state


def run_workers(structures, computer, *, batch=None, spawn=False, archive=False, directory="MyWorker") -> bool:
    """Run workers or a single worker chain through the legacy runtime."""
    for handler in list(logging.root.handlers):
        if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
            logging.root.removeHandler(handler)
    directory = pathlib.Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if not structures:
        raise ValueError("At least one structure is required.")
    if isinstance(structures[0], (str, pathlib.Path)):
        frames = []
        for index, source in enumerate(structures):
            builder = canonicalise_builder(str(source))
            builder.directory = directory / "init" / f"s{index}"
            frames.extend(builder.run())
    else:
        if not isinstance(structures[0], Atoms):
            raise TypeError(f"Unsupported structure type {type(structures[0]).__name__}.")
        frames = structures

    if not isinstance(computer, list) and hasattr(computer, "value"):
        computer = computer.value
    states = []
    if isinstance(computer, list) and (not computer or not isinstance(computer[0], list)):
        for index, worker in enumerate(computer):
            worker_directory = directory if len(computer) == 1 else directory / f"w{index}"
            states.append(run_one_worker(frames, worker, worker_directory, batch, spawn, archive))
    elif isinstance(computer, list) and computer and isinstance(computer[0], list):
        if len(computer) != 1:
            raise ValueError(f"Only one worker chain is supported, got {len(computer)}.")
        current_frames = frames
        for index, worker in enumerate(computer[0]):
            step_directory = directory / f"chainstep.{index:02d}"
            state = run_one_worker(current_frames, worker, step_directory, batch, spawn, archive)
            states.append(state)
            if state is not CompState.FINISHED:
                break
            current_frames = read(step_directory / "results" / "end_frames.xyz", ":")
            if index + 1 == len(computer[0]):
                result_link = directory / "results"
                if not result_link.exists():
                    result_link.symlink_to((step_directory / "results").relative_to(directory))
    else:
        raise TypeError(f"Unsupported computer {type(computer).__name__}.")
    return bool(states) and all(state is CompState.FINISHED for state in states)


# Compatibility names used by older callers.
convert_input_to_computer = create_computer
run_worker = run_workers
