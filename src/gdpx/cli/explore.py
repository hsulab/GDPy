#!/usr/bin/env python3
# -*- coding: utf-8 -*


import pathlib
from typing import Optional, Union

from gdpx import config
from gdpx.exploration.output import exploration_output
from gdpx.exploration.layout import exploration_layout, reject_legacy_layout
from gdpx.workflow.factory import create_exploration
from gdpx.execution.schedulers.factory import canonicalise_scheduler
from gdpx.execution.factory import create_worker
from gdpx.execution.workers.explore import ExplorationBasedWorker, run_exploration_in_commandline


def run_exploration(
    exp_params: dict,
    wait: Optional[float] = None,
    directory: Union[str, pathlib.Path] = "./",
    runtime=None,
    spawn: Optional[str] = None,
    *,
    input_path=None,
    random_seed=None,
):
    """Run an exploration.

    Args:
        exp_params: Exploration parameters.
        wait: Time to wait between runs. Defaults to None.
        directory: Directory for the exploration. Defaults to "./".
        runtime: Optional schema-v3 runtime. If omitted, use ``exp_params.runtime``.
        spawn: Comma-separated indices of explorations to run in commandline. Defaults to None.
        input_path: Optional input filename for the run header.
        random_seed: Optional effective global seed for the run header.

    """
    directory = pathlib.Path(directory)

    with exploration_output(directory, input_path, random_seed) as report:
        if runtime is not None:
            runtime_params = runtime
        else:
            if "runtime" in exp_params:
                runtime_params = exp_params.pop("runtime")
            else:
                raise RuntimeError("Exploration requires a runtime.")

        # Pop scheduler as exploration does not have it as an argument
        scheduler_params = exp_params.pop("scheduler", {})
        scheduler = canonicalise_scheduler(scheduler_params)

        # Create explorations
        exploration = create_exploration(exp_params)
        if isinstance(exploration, list):
            ...
        else:
            exploration = [exploration]

        report.start(len(exploration), scheduler.name)

        for curr_exploration in exploration:
            if hasattr(curr_exploration, "register_worker"):
                curr_exploration.register_worker(create_worker(runtime_params))

        num_explorations = len(exploration)
        if spawn:  # Run exploration in commandline as input files are prepared by worker
            exp_indices = [int(index) for index in spawn.split(",")]
            if any(index < 0 for index in exp_indices) or len(set(exp_indices)) != len(exp_indices):
                raise ValueError("Spawn indices must be distinct nonnegative integers.")
            reject_legacy_layout(directory)
            num_indices = len(exp_indices)
            assert (
                num_explorations == num_indices
            ), f"The numbers of explorations `{num_explorations}` and indices `{num_indices}` are not consistent."
            if num_explorations == 1:
                run_exploration_in_commandline(
                    wdir=directory,
                    exploration=exploration[0],
                    timewait=wait,
                    print_func=config._print,
                )
            else:
                directories = exploration_layout(directory)
                if directories is None:
                    directories = exploration_layout(
                        directory, max(num_explorations, max(exp_indices) + 1), create=True
                    )
                if max(exp_indices) >= len(directories):
                    raise ValueError("Spawn index is outside the saved exploration layout.")
                for i, exp in zip(exp_indices, exploration):
                    run_exploration_in_commandline(
                        directory / directories[i],
                        exp,
                        timewait=wait,
                        print_func=config._print,
                    )
            report.pending = sum(not exp.read_convergence() for exp in exploration)
        else:
            worker = ExplorationBasedWorker(exploration=exploration, scheduler=scheduler, directory=directory)
            worker.run()
            worker.inspect(resubmit=True)
            finished = {job.group_number for job in worker.job_store.get_finished()}
            report.pending = sum(index not in finished for index in range(num_explorations))

    return


if __name__ == "__main__":
    ...
