#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import enum
import dataclasses
import logging
import pathlib
from typing import Optional, Union

from ase import Atoms
from ase.io import read, write

from gdpx import config
from gdpx.factory.builder import canonicalise_builder
from gdpx.factory.computer import create_worker_chains, create_workers
from gdpx.factory.scheduler import canonicalise_scheduler
from gdpx.reactor.reactor import BaseReactor
from gdpx.utils.parser import parse_input_file
from gdpx.worker.drive import DriverBasedWorker
from gdpx.worker.grid import GridDriverBasedWorker

DEFAULT_MAIN_DIRNAME = "MyWorker"


def __getattr__(name):
    """Lazy compatibility exports for callers that still use workflow Variables."""
    if name in {"ComputerVariable", "ComputerChainVariable"}:
        from gdpx.nodes import computer as computer_nodes

        return getattr(computer_nodes, name)
    raise AttributeError(name)


CompState = enum.Enum("CompState", ("QUEUED", "FINISHED"))


def convert_input_to_computer(config):
    """Convert an input configuration to a computer.

    The `computer` can be ComputerVariable and ComputerChainVariable.
    This function should only be called in the `main.py`.

    """
    if isinstance(config, str) or isinstance(config, pathlib.Path):
        config = parse_input_file(input_fpath=config)

    if isinstance(config, dict):
        computer = convert_config_to_computer(config)
    elif isinstance(config, list):
        assert len(config) >= 1, "ComputerChain must have more than one computer configuration."
        computer = convert_config_to_computer_chain(config)
    else:
        raise RuntimeError(f"Unknown input for computer with a type of {config}.")

    return computer


def convert_config_to_computer_chain(config: list):
    """Create worker-major chains from computer configurations."""
    return create_worker_chains(config) if len(config) > 1 else create_workers(config[0])


def convert_config_to_computer(config):
    """Convert a configuration file or a dict to a computer."""
    if isinstance(config, dict):
        params = config
    else:  # assume it is json or yaml
        params = parse_input_file(input_fpath=config)

    assert isinstance(params, dict)

    # For compatibility
    potter_params = params.pop("potter", None)
    potential_params = params.pop("potential", None)
    if potter_params is None:
        if potential_params is not None:
            params["potter"] = potential_params
        else:
            raise RuntimeError("Fail to find any potter (potential) definition.")
    else:
        params["potter"] = potter_params

    return create_workers(params)


def run_one_worker(structures, worker, directory, batch, spawn, archive):
    """Run one worker on several structures."""
    # update working directory
    worker.directory = directory

    worker.is_spawned = spawn

    # run computations
    _ = worker.run(structures, batch=batch)

    # check results
    comp_state = CompState.QUEUED
    if not spawn:
        worker.inspect(resubmit=True, batch=batch)
        if worker.get_number_of_running_jobs() == 0:
            res_dir = directory / "results"
            if not res_dir.exists():
                res_dir.mkdir()
                ret = worker.retrieve(include_retrieved=True, use_archive=archive)
                if not isinstance(worker.driver, BaseReactor):
                    end_frames = [traj[-1] for traj in ret]
                    write(res_dir / "end_frames.xyz", end_frames)
                    config._print(f"{directory.name} has already been retrieved.")
                    comp_state = CompState.FINISHED
                else:
                    config._print("Reactor results cannot be retreived for now.")
            else:
                config._print(f"{directory.name} has already been retrieved.")
                comp_state = CompState.FINISHED

    return comp_state


def run_worker(
    structure: list[str],
    computer,
    *,
    batch: Optional[int] = None,
    spawn: bool = False,
    archive: bool = False,
    directory: Union[str, pathlib.Path] = pathlib.Path.cwd() / DEFAULT_MAIN_DIRNAME,
):
    """This computation is performed either by Computer or ComputerChain."""
    # some imported packages change `logging.basicConfig`
    # and accidently add a StreamHandler to logging.root
    # so remove it...
    for h in logging.root.handlers:
        if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler):
            logging.root.removeHandler(h)

    # Set working directory
    directory = pathlib.Path(directory)
    if not directory.exists():
        directory.mkdir()

    # Read structures
    if isinstance(structure[0], str):
        frames = []
        for i, s in enumerate(structure):
            builder = canonicalise_builder(s)
            builder.directory = directory / "init" / f"s{i}"
            frames.extend(builder.run())
    else:
        assert isinstance(structure[0], Atoms)
        frames = structure

    # Compatibility with legacy ComputerVariable/ComputerChainVariable inputs.
    if not isinstance(computer, list) and hasattr(computer, "value"):
        computer = computer.value

    # Find input frames
    comp_states = []
    if isinstance(computer, list) and (not computer or not isinstance(computer[0], list)):
        workers: list[DriverBasedWorker] = computer
        num_workers = len(workers)
        if num_workers == 1:
            comp_state = run_one_worker(frames, workers[0], directory, batch, spawn, archive)
            comp_states.append(comp_state)
        else:
            for i, w in enumerate(workers):
                comp_state = run_one_worker(frames, w, directory / f"w{i}", batch, spawn, archive)
                comp_states.append(comp_state)
    elif isinstance(computer, list) and computer and isinstance(computer[0], list):
        chains: list[DriverBasedWorker] = computer
        num_chains = len(chains)
        if num_chains != 1:  # TODO: Unify the code below with compute_chain operation
            raise Exception(f"ComputerChain supports only one chain for now, but got {num_chains} chains.")
        workers = chains[0]
        num_workers = len(workers)
        curr_frames = frames
        for i, worker in enumerate(workers):
            config._print(f"<- ComputerChainStep.{str(i).zfill(2)} ->")
            chainstep_directory = directory / f"chainstep.{str(i).zfill(2)}"
            comp_state = run_one_worker(curr_frames, worker, chainstep_directory, batch, spawn, archive)
            if comp_state == CompState.FINISHED:
                config._print("chainstep is finished.")
                curr_frames = read(chainstep_directory / "results" / "end_frames.xyz", ":")
                # link to the results from the last chainstep
                if i + 1 == num_workers:
                    (directory / "results").symlink_to((chainstep_directory / "results").relative_to(directory))
            else:
                config._print("chainstep is not finished.")
                break
    else:
        raise RuntimeError(f"Unknown computer `{computer}`.")

    # Check computation states
    is_finished = False
    if all([comp_state == CompState.FINISHED for comp_state in comp_states]):
        is_finished = True

    return is_finished


def convert_config_to_grid_components(grid_params: dict):
    """"""
    # config._print(grid_params)
    grid_data = grid_params.get("grid", None)
    assert grid_data is not None

    # scheduler
    scheduler = canonicalise_scheduler(grid_params.get("scheduler", {}))

    structures, potters, drivers = [], [], []
    for data in grid_data:
        builder_params = data.get("builder")
        builder = canonicalise_builder(builder_params)
        structures.extend(builder.run())

        # FIXME: broadcast worker to structures?
        computer_params = data.get("computer")
        computers = create_workers(computer_params)
        num_computers = len(computers)
        assert num_computers == 1
        potters.append(computers[0].potter)
        drivers.append(computers[0].driver)
        # config._print(f"{builder =}")
        # config._print(f"{computer =}")
        ...
    # config._print(f"{structures =}")
    # config._print(f"{potters =}")
    # config._print(f"{drivers =}")

    # aux parameters
    batchsize = grid_params.get("batchsize", 1)

    aux_params = dict(batchsize=batchsize)

    return scheduler, structures, potters, drivers, aux_params


def run_grid_worker(grid_params: dict, batch: Optional[int], spawn, directory):
    """"""
    directory = pathlib.Path(directory)

    if batch is None:  # submit jobs to queue
        scheduler, structures, potters, drivers, aux_params = convert_config_to_grid_components(grid_params)
        worker = GridDriverBasedWorker(potters=potters, drivers=drivers, scheduler=scheduler, **aux_params)
        worker._submit = False

        # run computations
        worker.driver = None  # FIXME: compat
        run_one_worker(
            structures,
            worker,
            directory,
            batch=batch,
            spawn=spawn,
            archive=True,
        )
    else:  # run jobs in command line
        batch_grid_params, batch_wdirs = [], []
        for x in grid_params["grid"]:
            if x["batch"] == batch:
                batch_grid_params.append(x)
                batch_wdirs.append(directory / x["wdir_name"])
        grid_params = {}
        grid_params["grid"] = batch_grid_params

        scheduler, structures, potters, drivers, aux_params = convert_config_to_grid_components(grid_params)
        # aux_params["batchsize"] = len(structures)
        # worker = GridDriverBasedWorker(
        #     potters=potters, drivers=drivers, scheduler=scheduler, **aux_params
        # )
        GridDriverBasedWorker.run_grid_computations_in_command(
            batch_wdirs, structures, drivers, print_func=config._print
        )

    return


def run_computation(
    structure,
    computer,
    *,
    batch: Optional[int] = None,
    spawn: bool = False,
    archive: bool = False,
    directory: Union[str, pathlib.Path] = pathlib.Path.cwd() / DEFAULT_MAIN_DIRNAME,
    plan: Optional[Union[str, pathlib.Path]] = None,
    worker_index: int = 0,
):
    """Run a legacy computation or one explicit lifecycle action."""
    lifecycle_actions = {"prepare", "submit", "run", "status", "resubmit", "collect"}
    action = structure[0] if structure and structure[0] in lifecycle_actions else None

    if action is not None:
        from gdpx.compute import (
            collect_compute,
            inspect_compute,
            prepare_compute,
            resubmit_compute,
            run_compute_batch,
            submit_compute,
        )

        plan_path = pathlib.Path(plan) if plan is not None else pathlib.Path(directory)
        if action == "prepare":
            if computer is None:
                raise RuntimeError("`gdp compute prepare` requires `-p/--potential` with a worker configuration.")
            result = prepare_compute(computer, structure[1:], directory)
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
        else:
            result = collect_compute(plan_path, archive=archive)
        config._print(dataclasses.asdict(result))
        return result

    if computer is not None:
        if spawn:
            # Compatibility for scheduler scripts generated before compute plans.
            computer_obj = convert_input_to_computer(computer)
            return run_worker(
                structure,
                computer_obj,
                batch=batch,
                spawn=True,
                archive=archive,
                directory=directory,
            )

        parsed = parse_input_file(computer) if isinstance(computer, (str, pathlib.Path)) else computer
        if isinstance(parsed, dict):
            from gdpx.compute import orchestrate_compute, prepare_compute

            compute_plan = prepare_compute(parsed, structure, directory)
            result = orchestrate_compute(compute_plan, archive=archive)
            config._print(dataclasses.asdict(result))
            return result

        # Computer chains retain their legacy implementation in the first slice.
        computer_obj = convert_input_to_computer(parsed)
        return run_worker(
            structure,
            computer_obj,
            batch=batch,
            spawn=False,
            archive=archive,
            directory=directory,
        )
    else:
        # Use GridWorker here!!
        run_grid_worker(
            parse_input_file(structure[0]),
            batch=batch,
            spawn=spawn,
            directory=directory,
        )

    return None


if __name__ == "__main__":
    ...
