#!/usr/bin/env python3
# -*- coding: utf-8 -*


import pathlib
import time

from typing import Optional


from .. import config
from ..scheduler.interface import SchedulerVariable
from ..expedition.interface import ExpeditionVariable
from ..worker.explore import ExpeditionBasedWorker


def run_expedition(
    exp_params: dict, wait: Optional[float] = None, directory: str = "./", potter=None
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
    expedition.directory = directory
    if hasattr(expedition, "register_worker"):
        expedition.register_worker(exp_params["worker"])

    if scheduler.name == "local":
        if wait is not None:
            for i in range(1000):
                expedition.run()
                if expedition.read_convergence():
                    break
                time.sleep(wait)
                config._print(f"wait {wait} seconds...")
            else:
                ...
        else:
            expedition.run()
    else:  # submit to queue
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
