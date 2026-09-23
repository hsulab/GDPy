#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import copy
from typing import Union

from gdpx.scheduler import REGISTER as SCHEDULER_REGISTER
from gdpx.scheduler.scheduler import BaseScheduler


def canonicalise_scheduler(config: Union[dict, BaseScheduler]) -> BaseScheduler:
    """Cannonicalise the scheduler based on the input configuration."""
    if isinstance(config, dict):
        config = copy.deepcopy(config)
        backend = config.pop("backend", "local").capitalize() + "Scheduler"
        if backend in SCHEDULER_REGISTER:
            scheduler = SCHEDULER_REGISTER[backend](**config)
        else:
            raise Exception(f"Scheduler backend `{backend}` is not registered.")
    else:
        # Assume the config is already a BaseScheduler instance
        scheduler = copy.deepcopy(config)

    return scheduler


if __name__ == "__main__":
    ...
