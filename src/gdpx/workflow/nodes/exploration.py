#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import copy
import pathlib
from collections.abc import Mapping
from typing import Union

import omegaconf

from gdpx.workflow.session.registry import workflow_registers as registers
from gdpx.workflow.factory import create_exploration
from gdpx.exploration.exploration import BaseExploration
from gdpx.exploration.layout import exploration_layout
from gdpx.workflow.session.operation import Operation
from gdpx.workflow.session.variable import DummyVariable, Variable
from gdpx.execution.workers.explore import ExplorationBasedWorker

from .scheduler import SchedulerVariable


@registers.variable.register
class ExplorationVariable(Variable):

    def __init__(self, directory: Union[str, pathlib.Path] = "./", **kwargs):
        """"""
        recipe = kwargs.get("recipe")
        if isinstance(recipe, Mapping) and isinstance(recipe.get("builder"), Variable):
            kwargs["recipe"] = copy.deepcopy(dict(recipe))
            kwargs["recipe"]["builder"] = recipe["builder"].value
        elif isinstance(recipe, Mapping) and isinstance(recipe.get("population"), Mapping):
            builders = recipe["population"].get("builders")
            if isinstance(builders, Mapping) and any(isinstance(builder, Variable) for builder in builders.values()):
                kwargs["recipe"] = copy.deepcopy(dict(recipe))
                kwargs["recipe"]["population"] = copy.deepcopy(dict(recipe["population"]))
                kwargs["recipe"]["population"]["builders"] = {
                    name: builder.value if isinstance(builder, Variable) else builder
                    for name, builder in builders.items()
                }
        if isinstance(kwargs.get("builder"), Variable):
            kwargs["builder"] = kwargs["builder"].value
        system = kwargs.get("system")
        if isinstance(system, Mapping) and isinstance(system.get("builder"), Variable):
            kwargs["system"] = copy.deepcopy(dict(system))
            kwargs["system"]["builder"] = system["builder"].value
        exploration = create_exploration(kwargs)

        super().__init__(initial_value=exploration, directory=directory)

        return

    @property
    def value(self) -> BaseExploration:
        """"""

        return self._value  # type: ignore

@registers.operation.register
class explore(Operation):

    #: Whether to actively update some attrs.
    _active: bool = False

    def __init__(
        self,
        exploration,
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

        input_nodes = [exploration, worker, scheduler]
        super().__init__(input_nodes, directory)

        self.wait_time = wait_time

        self._active = active

        return

    def forward(self, exploration, dyn_worker, scheduler):
        """Explore an exploration and forward results for further analysis.

        Returns:
            Workers that store structures.

        """
        super().forward()

        if isinstance(exploration, list):
            explorations = exploration
        else:
            explorations = [exploration]

        num_explorations = len(explorations)
        if self._active:
            curr_iter = int(self.directory.parent.name.split(".")[-1])
            if curr_iter > 0:
                self._print("    >>> Update seed_file...")
                previous = self.directory.parent.parent / f"iter.{curr_iter-1:04d}" / self.directory.name
                directories = exploration_layout(previous, num_explorations)
                for i, current in enumerate(explorations):
                    if hasattr(current, "update_active_params"):
                        current.update_active_params(previous / directories[i])

        self._print(f"{dyn_worker=}")
        for exploration in explorations:
            if hasattr(exploration, "register_worker"):
                exploration.register_worker(dyn_worker)

        worker = ExplorationBasedWorker(explorations, scheduler)
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
