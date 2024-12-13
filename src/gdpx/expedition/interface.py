#!/usr/bin/env python3
# -*- coding: utf-8 -*


import copy
import pathlib
from typing import Union, Iterable

import numpy as np

from ..core.operation import Operation
from ..core.register import registers
from ..core.variable import Variable
from ..scheduler.interface import SchedulerVariable
from ..worker.explore import ExpeditionBasedWorker
from .expedition import AbstractExpedition


def register_expedition_methods():
    """"""
    from .genetic_algorithm.mutation.exchange import ExchangeMutation

    registers.builder.register("exchange_mutation")(ExchangeMutation)

    from .genetic_algorithm.engine import GeneticAlgorithmBroadcaster

    registers.expedition.register("genetic_algorithm")(GeneticAlgorithmBroadcaster)

    from .monte_carlo.basin_hopping import BasinHopping

    registers.expedition.register("basin_hopping")(BasinHopping)

    from .monte_carlo.hybrid_monte_carlo import HybridMonteCarlo

    registers.expedition.register("hybrid_monte_carlo")(HybridMonteCarlo)

    from .monte_carlo.concurrent_hopping import ConcurrentHopping

    registers.expedition.register("concurrent_hopping")(ConcurrentHopping)

    from .monte_carlo.monte_carlo import MonteCarlo

    registers.expedition.register("monte_carlo")(MonteCarlo)

    from .simulated_annealing.simulated_annealing import SimulatedAnnealing

    registers.expedition.register("simulated_annealing")(SimulatedAnnealing)

    from .artificial_force.afir import AFIRSearch

    registers.expedition.register("artificial_reaction")(AFIRSearch)

    return


register_expedition_methods()


@registers.variable.register
class ExpeditionVariable(Variable):

    def __init__(self, directory: Union[str, pathlib.Path] = "./", **kwargs):
        """"""
        random_seed = kwargs.get("random_seed", None)
        if random_seed is None:
            random_seed = np.random.randint(0, 1e8)  # type: ignore

        method = kwargs.pop("method", None)
        if "builder" in kwargs:
            builder = self._canonicalise_builder(kwargs["builder"], random_seed)
            kwargs["builder"] = builder

        expedition = registers.create(
            "expedition", method, convert_name=False, **kwargs
        )
        if isinstance(expedition, Iterable):
            expedition = list(expedition)  # A List of expeditions

        super().__init__(initial_value=expedition, directory=directory)

        return

    @property
    def value(self) -> AbstractExpedition:
        """"""

        return self._value  # type: ignore

    def _canonicalise_builder(self, builder: dict, random_seed: int):
        """Canonicalise the builder and set random seed."""
        if builder is not None:
            if isinstance(builder, dict):
                builder_params = copy.deepcopy(builder)
                builder_method = builder_params.pop("method")
                builder = registers.create(
                    "builder", builder_method, convert_name=False, **builder_params
                )
            else:  # variable
                builder = builder.value
                np.random.seed(random_seed)
        else:
            builder = None

        return builder


@registers.operation.register
class explore(Operation):

    #: Whether to actively update some attrs.
    _active: bool = False

    def __init__(
        self,
        expedition,
        worker,
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
                        self.directory.parent.parent
                        / f"iter.{str(curr_iter-1).zfill(4)}"
                        / self.directory.name
                    ) / f"expedition-{i}"
                    if hasattr(expedition, "update_active_params"):
                        expedition.update_active_params(prev_wdir)

        self._print(f"{dyn_worker=}")
        for expedition in expeditions:
            if hasattr(expedition, "register_worker"):
                expedition.register_worker(dyn_worker)

        worker = ExpeditionBasedWorker(expeditions, scheduler)
        worker.directory = self.directory
        worker.wait_time = self.wait_time

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
