#!/usr/bin/env python3
# -*- coding: utf-8 -*


import copy
import logging
import pathlib
import time

import numpy as np

from .. import config
from ..utils.command import parse_input_file
from ..core.variable import Variable
from ..core.operation import Operation
from ..core.register import registers
from ..worker.explore import ExpeditionBasedWorker
from ..scheduler.interface import SchedulerVariable


"""
"""

class ExpeditionVariable(Variable):

    def __init__(self, directory="./", **kwargs):
        """"""
        random_seed = kwargs.get("random_seed", None)
        if random_seed is None:
            random_seed = np.random.randint(0, 1e8)

        method = kwargs.pop("method", None)
        if "builder" in kwargs:
            builder = self._create_a_builder(kwargs["builder"], random_seed)
            kwargs["builder"] = builder

        expedition = registers.create(
            "expedition", method, convert_name=False, **kwargs
        )

        super().__init__(initial_value=expedition, directory=directory)

        return
    
    def _create_a_builder(self, builder: dict, random_seed: int):
        """"""
        # - builder
        if isinstance(builder, dict):
            builder_params = copy.deepcopy(builder)
            builder_method = builder_params.pop("method")
            builder = registers.create(
                "builder", builder_method, convert_name=False, **builder_params
            )
        else:  # variable
            builder = builder.value
            np.random.seed(random_seed)
        
        return builder


class explore(Operation):

    #: Whether to actively update some attrs.
    _active: bool = False

    def __init__(
        self, expedition, worker, scheduler=None, wait_time=60, 
        active: bool=False, directory="./", *args, **kwargs
    ) -> None:
        """"""
        if scheduler is None:
            scheduler = SchedulerVariable()
        input_nodes = [expedition, worker, scheduler]
        super().__init__(input_nodes, directory)

        self.wait_time = wait_time

        self._active = active

        return

    def forward(self, expedition, dyn_worker, scheduler):
        """Explore an expedition and forward results for further analysis.

        Returns:
            Workers that store structures.
        
        """
        super().forward()

        expeditions = [expedition]
        nexpeditions = len(expeditions)
        if self._active:
            curr_iter = int(self.directory.parent.name.split(".")[-1])
            if curr_iter > 0:
                self._print("    >>> Update seed_file...")
                for i in range(nexpeditions):
                    prev_wdir = (
                        self.directory.parent.parent / 
                        f"iter.{str(curr_iter-1).zfill(4)}" / 
                        self.directory.name
                    ) / f"expedition-{i}"
                    if hasattr(expedition, "update_active_params"):
                        expedition.update_active_params(prev_wdir)

        # -
        if hasattr(expedition, "register_worker"):
            expedition.register_worker(dyn_worker)
        
        # - run expedition with a worker
        worker = ExpeditionBasedWorker(expedition, scheduler)
        worker.directory = self.directory
        worker.wait_time = self.wait_time

        worker.run()
        worker.inspect(resubmit=True)

        basic_workers = []
        if worker.get_number_of_running_jobs() == 0:
            basic_workers = worker.retrieve(include_retrieved=True)
            self._debug(f"basic_workers: {basic_workers}")
            # for w in basic_workers:
            #     print(w.directory)
            self.status = "finished"
        else:
            ...

        return basic_workers


def run_expedition(exp_params: dict, wait: float=None, directory="./", potter=None):
    """"""
    directory = pathlib.Path(directory)

    if potter is not None:
        exp_params["worker"] = potter
    else:
        if "worker" not in exp_params:
            raise RuntimeError("Expedition must have a worker.")

    scheduler_params = exp_params.pop("scheduler", {})
    scheduler = SchedulerVariable(**scheduler_params).value

    #method = exp_params.pop("method")
    #expedition = registers.create("variable", method, convert_name=True, **exp_params).value
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
    else: # submit to queue
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
