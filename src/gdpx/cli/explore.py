#!/usr/bin/env python3
# -*- coding: utf-8 -*


import pathlib
from typing import Optional, Union

from gdpx import config
from gdpx.exploration.output import exploration_output
from gdpx.workflow.factory import create_expedition
from gdpx.execution.schedulers.factory import canonicalise_scheduler
from gdpx.execution.factory import create_worker
from gdpx.execution.workers.explore import ExpeditionBasedWorker, run_expedition_in_commandline


def run_expedition(
    exp_params: dict,
    wait: Optional[float] = None,
    directory: Union[str, pathlib.Path] = "./",
    runtime=None,
    spawn: Optional[str] = None,
    *,
    input_path=None,
    random_seed=None,
):
    """Run an expedition.

    Args:
        exp_params: Expedition parameters.
        wait: Time to wait between runs. Defaults to None.
        directory: Directory for the expedition. Defaults to "./".
        runtime: Optional schema-v3 runtime. If omitted, use ``exp_params.runtime``.
        spawn: Comma-separated indices of expeditions to run in commandline. Defaults to None.
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

        # Pop scheduler as expedition does not have it as an argument
        scheduler_params = exp_params.pop("scheduler", {})
        scheduler = canonicalise_scheduler(scheduler_params)

        # Create expeditions
        expedition = create_expedition(exp_params)
        if isinstance(expedition, list):
            ...
        else:
            expedition = [expedition]

        report.start(len(expedition), scheduler.name)

        for curr_expedition in expedition:
            if hasattr(curr_expedition, "register_worker"):
                curr_expedition.register_worker(create_worker(runtime_params))

        num_expeditions = len(expedition)
        if spawn:  # Run expedition in commandline as input files are prepared by worker
            exp_indices = spawn.split(",")
            num_indices = len(exp_indices)
            assert (
                num_expeditions == num_indices
            ), f"The numbers of expeditiosn `{num_expeditions}` and indices `{num_indices}` are not consistent."
            if num_expeditions == 1:
                run_expedition_in_commandline(
                    wdir=directory,
                    expedition=expedition[0],
                    timewait=wait,
                    print_func=config._print,
                )
            else:
                for i, exp in zip(exp_indices, expedition):
                    run_expedition_in_commandline(
                        directory / f"expedition-{i}",
                        exp,
                        timewait=wait,
                        print_func=config._print,
                    )
            report.pending = sum(not exp.read_convergence() for exp in expedition)
        else:
            worker = ExpeditionBasedWorker(expedition=expedition, scheduler=scheduler, directory=directory)
            worker.run()
            worker.inspect(resubmit=True)
            finished = {job.group_number for job in worker.job_store.get_finished()}
            report.pending = sum(index not in finished for index in range(num_expeditions))

    return


if __name__ == "__main__":
    ...
