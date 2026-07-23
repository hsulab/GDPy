#!/usr/bin/env python3
# -*- coding: utf-8 -*


import pathlib
from typing import Optional, Union

from gdpx import config
from gdpx.factory.components import create_expedition
from gdpx.factory.scheduler import canonicalise_scheduler
from gdpx.worker.explore import ExpeditionBasedWorker, run_expedition_in_commandline


def run_expedition(
    exp_params: dict,
    wait: Optional[float] = None,
    directory: Union[str, pathlib.Path] = "./",
    potter=None,
    spawn: Optional[str] = None,
):
    """Run an expedition.

    Args:
        exp_params: Expedition parameters.
        wait: Time to wait between runs. Defaults to None.
        directory: Directory for the expedition. Defaults to "./".
        potter: Optional worker parameters. If None, it will be taken from `exp_params`.
        spawn: Comma-separated indices of expeditions to run in commandline. Defaults to None.

    """
    directory = pathlib.Path(directory)

    if potter is not None:
        worker_params = potter
    else:
        if "worker" in exp_params:
            worker_params = exp_params.pop("worker")
        else:
            raise RuntimeError("Expedition must have a worker.")

    # Pop scheduler as expedition does not have it as an argument
    scheduler_params = exp_params.pop("scheduler", {})
    scheduler = canonicalise_scheduler(scheduler_params)

    # Create expeditions
    expedition = create_expedition(exp_params)
    if isinstance(expedition, list):
        ...
    else:
        expedition = [expedition]

    for curr_expedition in expedition:
        if hasattr(curr_expedition, "register_worker"):
            curr_expedition.register_worker(worker_params)

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
    else:
        worker = ExpeditionBasedWorker(expedition=expedition, scheduler=scheduler, directory=directory)
        worker.run()
        worker.inspect(resubmit=True)
        if worker.get_number_of_running_jobs() == 0:
            config._print("Expedition finished...")
        else:
            ...

    return


if __name__ == "__main__":
    ...
