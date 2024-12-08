#!/usr/bin/env python3
# -*- coding: utf-8 -*


import pathlib
from typing import Optional, Union, Iterable

from .. import config
from ..expedition.interface import ExpeditionVariable
from ..scheduler.interface import SchedulerVariable
from ..utils.logio import remove_extra_stream_handlers
from ..worker.explore import ExpeditionBasedWorker


def run_expedition(
    exp_params: dict,
    wait: Optional[float] = None,
    directory: Union[str, pathlib.Path] = "./",
    potter=None,
):
    """"""
    directory = pathlib.Path(directory)

    if potter is not None:
        exp_params["worker"] = potter
    else:
        if "worker" not in exp_params:
            raise RuntimeError("Expedition must have a worker.")

    scheduler_params = exp_params.pop("scheduler", {})
    scheduler = SchedulerVariable(**scheduler_params).value

    expedition = ExpeditionVariable(directory=directory, **exp_params).value
    if isinstance(expedition, Iterable):
        for expd in expedition:
            if hasattr(expd, "register_worker"):
                expd.register_worker(exp_params["worker"])
    else:
        if hasattr(expedition, "register_worker"):
            expedition.register_worker(exp_params["worker"])

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
