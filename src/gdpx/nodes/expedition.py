#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import copy
import pathlib
from typing import Union

import omegaconf

from gdpx.session.registry import workflow_registers as registers
from gdpx.factory.components import create_expedition
from gdpx.expedition.expedition import BaseExpedition
from gdpx.session.operation import Operation
from gdpx.session.variable import DummyVariable, Variable
from gdpx.worker.explore import ExpeditionBasedWorker

from .scheduler import SchedulerVariable


@registers.variable.register
class ExpeditionVariable(Variable):

    def __init__(self, directory: Union[str, pathlib.Path] = "./", **kwargs):
        """"""
        if isinstance(kwargs.get("builder"), Variable):
            kwargs["builder"] = kwargs["builder"].value
        expedition = create_expedition(kwargs)

        super().__init__(initial_value=expedition, directory=directory)

        return

    @property
    def value(self) -> BaseExpedition:
        """"""

        return self._value  # type: ignore

@registers.operation.register
class explore(Operation):

    #: Whether to actively update some attrs.
    _active: bool = False

    def __init__(
        self,
        expedition,
        worker=DummyVariable(),
        scheduler=None,
        wait_time=60,
        active: bool = False,
        directory="./",
        *args,
        **kwargs,
    ) -> None:
        """"""
        if scheduler is None:
            scheduler = SchedulerVariable()
        if isinstance(scheduler, dict) or isinstance(scheduler, omegaconf.DictConfig):
            scheduler_params = copy.deepcopy(scheduler)
            scheduler = SchedulerVariable(**scheduler_params)
        elif isinstance(scheduler, Variable):
            ...
        else:
            raise Exception(f"Unknown {scheduler} for the scheduler.")

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

        if isinstance(expedition, list):
            expeditions = expedition
        else:
            expeditions = [expedition]

        num_expeditions = len(expeditions)
        if self._active:
            curr_iter = int(self.directory.parent.name.split(".")[-1])
            if curr_iter > 0:
                self._print("    >>> Update seed_file...")
                for i in range(num_expeditions):
                    prev_wdir = (
                        self.directory.parent.parent / f"iter.{str(curr_iter-1).zfill(4)}" / self.directory.name
                    ) / f"expedition-{i}"
                    if hasattr(expedition, "update_active_params"):
                        expedition.update_active_params(prev_wdir)

        self._print(f"{dyn_worker=}")
        for expedition in expeditions:
            if hasattr(expedition, "register_worker"):
                expedition.register_worker(dyn_worker)

        worker = ExpeditionBasedWorker(expeditions, scheduler)
        worker.directory = self.directory
        worker.timewait = self.wait_time

        worker.run()
        worker.inspect(resubmit=True)

        if worker.get_number_of_running_jobs() == 0:
            basic_workers = worker.retrieve(include_retrieved=True)
            self._debug(f"basic_workers: {basic_workers}")
            self.status = "finished"
        else:
            basic_workers = []

        return basic_workers


if __name__ == "__main__":
    ...
