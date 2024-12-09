#!/usr/bin/env python3
# -*- coding: utf-8 -*


import pathlib
from typing import Iterable, Optional, Union

from numpy import expand_dims

from .. import config
from ..expedition.interface import ExpeditionVariable
from ..scheduler.interface import SchedulerVariable
from ..utils.logio import remove_extra_stream_handlers
from ..worker.explore import ExpeditionBasedWorker, run_expedition_in_commandline


def run_expedition(
    exp_params: dict,
    wait: Optional[float] = None,
    directory: Union[str, pathlib.Path] = "./",
    potter=None,
    spawn: Optional[str] = None,
):
    """"""
    directory = pathlib.Path(directory)

    if potter is not None:
        exp_params["worker"] = potter
    else:
        if "worker" not in exp_params:
            raise RuntimeError("Expedition must have a worker.")

    # Pop scheduler as expedition does not have it as an argument
    scheduler_params = exp_params.pop("scheduler", {})
    scheduler = SchedulerVariable(**scheduler_params).value

    # Create expeditions
    expedition = ExpeditionVariable(directory=directory, **exp_params).value
    if isinstance(expedition, list):
        ...
    else:
        expedition = [expedition]

    for curr_expedition in expedition:
        if hasattr(curr_expedition, "register_worker"):
            curr_expedition.register_worker(exp_params["worker"])

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
                timewait=None,
                print_func=config._print,
            )
        else:
            for i, exp in zip(exp_indices, expedition):
                run_expedition_in_commandline(
                    directory / f"expedition-{i}",
                    exp,
                    timewait=None,
                    print_func=config._print,
                )
    else:
        remove_extra_stream_handlers()

        worker = ExpeditionBasedWorker(
            expedition=expedition, scheduler=scheduler, directory=directory
        )
        worker.run()
        worker.inspect(resubmit=True)
        if worker.get_number_of_running_jobs() == 0:
            config._print("Expedition finished...")
        else:
            ...

    return


if __name__ == "__main__":
    ...
