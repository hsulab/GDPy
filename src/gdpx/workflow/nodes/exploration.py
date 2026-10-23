#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import copy
import pathlib
from collections.abc import Mapping
from typing import Union

from gdpx.execution.factory import create_worker
from gdpx.execution.runtime import Runtime
from gdpx.execution.workers.explore import ExplorationBasedWorker
from gdpx.exploration.exploration import BaseExploration
from gdpx.workflow.factory import create_exploration
from gdpx.workflow.session.operation import Operation
from gdpx.workflow.session.registry import workflow_registers as registers
from gdpx.workflow.session.variable import DummyVariable, Variable
from gdpx.workflow.state import NamedOutputs

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
        elif isinstance(system, Mapping) and isinstance(system.get("builders"), Mapping):
            builders = system["builders"]
            if any(isinstance(builder, Variable) for builder in builders.values()):
                kwargs["system"] = copy.deepcopy(dict(system))
                kwargs["system"]["builders"] = {
                    name: builder.value if isinstance(builder, Variable) else builder
                    for name, builder in builders.items()
                }
        exploration = create_exploration(kwargs)

        super().__init__(initial_value=exploration, directory=directory)

        return

    @property
    def value(self) -> BaseExploration:
        """"""

        return self._value  # type: ignore

@registers.operation.register
class explore(Operation):
    output_names = ("minimised_candidates", "dynamics", "minimisation", "continuation")
    default_output = "minimised_candidates"

    def __init__(
        self,
        exploration,
        worker=DummyVariable(),
        runtime=None,
        scheduler=None,
        continuation=DummyVariable(),
        wait_time=60,
        directory="./",
        *args,
        **kwargs,
    ) -> None:
        """"""
        if scheduler is None:
            scheduler = SchedulerVariable()
        if isinstance(scheduler, Mapping):
            scheduler_params = copy.deepcopy(scheduler)
            scheduler = SchedulerVariable(**scheduler_params)
        elif isinstance(scheduler, Variable):
            ...
        else:
            raise Exception(f"Unknown {scheduler} for the scheduler.")

        if runtime is not None:
            if not isinstance(worker, DummyVariable):
                raise ValueError("explore accepts either runtime or worker, not both.")
            worker = runtime
        input_nodes = [exploration, worker, scheduler, continuation]
        self.builder_names = []
        for key, value in kwargs.items():
            if not key.startswith("builder_") or not key[8:]:
                raise TypeError(f"Unknown explore input {key!r}; use builder_<name> for structure bindings.")
            self.builder_names.append(key[8:])
            input_nodes.append(value)
        super().__init__(input_nodes, directory)

        self.wait_time = wait_time

        return

    def forward(self, exploration, dyn_worker, scheduler, continuation=None, *builder_structures):
        """Explore an exploration and forward results for further analysis.

        Returns:
            Workers that store structures.

        """
        super().forward()

        if isinstance(exploration, list):
            explorations = exploration
        else:
            explorations = [exploration]

        if continuation is None:
            continuations = [None] * len(explorations)
        elif isinstance(continuation, (list, tuple)):
            continuations = list(continuation)
        else:
            continuations = [continuation]
        if len(continuations) != len(explorations):
            raise ValueError("Exploration continuation count does not match explorations.")
        if len(builder_structures) != len(self.builder_names):
            raise ValueError("Exploration builder binding count does not match inputs.")
        for index, (current, previous) in enumerate(zip(explorations, continuations)):
            for name, structures in zip(self.builder_names, builder_structures):
                if structures is not None:
                    if previous is not None:
                        raise ValueError("Builder bindings require a fresh exploration, without continuation.")
                    if not hasattr(current, "bind_initial_builder"):
                        raise TypeError(f"{type(current).__name__} does not support initial builder bindings.")
                    current.bind_initial_builder(name, structures, self.directory / "builder_inputs" / str(index))
            if hasattr(current, "restore_continuation"):
                current.restore_continuation(previous)
            elif previous is not None:
                raise TypeError(f"{type(current).__name__} does not support workflow continuation.")

        if isinstance(dyn_worker, (Runtime, Mapping)):
            dyn_worker = create_worker(dyn_worker)
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
            return NamedOutputs(minimised_candidates=[], dynamics=[], minimisation=[], continuation=continuation)

        next_continuations = tuple(
            current.capture_continuation()
            if hasattr(current, "capture_continuation")
            else None
            for current in explorations
        )
        continuation_output = next_continuations[0] if len(next_continuations) == 1 else next_continuations
        def workers_for(method):
            return [worker for worker in basic_workers
                    if getattr(getattr(worker, "runtime", None), "config", None) is not None
                    and worker.runtime.config.executor.method == method]

        candidates = []
        for current in explorations:
            if hasattr(current, "get_minimised_candidates"):
                candidates.extend(current.get_minimised_candidates())
            elif hasattr(current, "get_workers"):
                # Explorations without a candidate database can expose the
                # successful endpoints of their minimisation calculations.
                for basic in current.get_workers():
                    runtime = getattr(basic, "runtime", None)
                    if runtime is not None and runtime.config.executor.method == "min":
                        candidates.extend(trajectory[-1] for trajectory in
                                          basic.retrieve(include_retrieved=True, use_archive=False)
                                          if trajectory and not trajectory[-1].info.get("earlystop"))
        return NamedOutputs(minimised_candidates=candidates,
                            dynamics=workers_for("md"), minimisation=workers_for("min"),
                            continuation=continuation_output)


if __name__ == "__main__":
    ...
