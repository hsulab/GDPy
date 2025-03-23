#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import copy
import itertools
import pathlib
from typing import Callable, Optional

import omegaconf
from ase.calculators.calculator import BaseCalculator

from gdpx.computation.driver import BaseDriver
from gdpx.core.register import registers
from gdpx.potential.manager import BasePotentialManager
from gdpx.potential.utils import convert_input_to_potter
from gdpx.reactor.reactor import BaseReactor
from gdpx.scheduler.scheduler import BaseScheduler
from gdpx.session.variable import Variable
from gdpx.utils.parser import parse_input_file
from gdpx.worker.drive import DriverBasedWorker
from gdpx.worker.grid import GridDriverBasedWorker
from gdpx.worker.react import ReactorBasedWorker
from gdpx.worker.single import SingleWorker
from gdpx.worker.worker import BaseWorker


def broadcast_and_adjust_potter(
    inp,
    estimate_uncertainty: Optional[bool] = False,
    switch_backend: Optional[str] = None,
    print_func: Callable = print,
) -> list[BasePotentialManager]:
    """Convert an input to a potter and adjust its behaviour."""
    # Convert anything into potter
    potter = convert_input_to_potter(inp)
    assert isinstance(potter, BasePotentialManager), f"{potter} is not a `BasePotentialManager` but `{type(potter)}`."

    # HACK: broadcast potters
    if hasattr(potter, "broadcast"):
        potters = potter.broadcast(potter)
    else:
        potters = [potter]

    for p in potters:
        assert isinstance(p.calc, BaseCalculator), f"{p.calc} is not `Calculator`."

    # Adjust potter behaviour
    for i, potter in enumerate(potters):
        print_func(f"potter-{i} {potter.name}")
        if hasattr(potter, "switch_uncertainty_estimation"):
            if estimate_uncertainty is not None:
                print_func(f"{potter.name} switches its uncertainty estimation to {estimate_uncertainty}...")
                potter.switch_uncertainty_estimation(estimate_uncertainty)
            else:
                ...
        else:
            print_func(f"{potter.name} does not support switching its uncertainty estimation...")

        if hasattr(potter, "switch_backend"):
            if switch_backend is not None:
                print_func(f"{potter.name} switches its backend to {switch_backend}...")
                potter.switch_backend(backend=switch_backend)
            else:
                ...
        else:
            print_func(f"{potter.name} does not support switching its backend...")

    return potters


@registers.variable.register
class ComputerChainVariable(Variable):

    def __init__(self, computers, directory=pathlib.Path.cwd()):
        """"""
        value = self._canonicalise_input_nodes([computers])
        super().__init__(value)

        self._init_params = copy.deepcopy([[w.as_dict() for w in c] for c in value])

        return

    def _canonicalise_input_nodes(self, input_nodes):
        """"""
        (computers,) = input_nodes

        if isinstance(computers, list) or isinstance(computers, omegaconf.ListConfig):
            computers_ = []
            for computer in computers:
                if isinstance(computer, str) or isinstance(computer, pathlib.Path):
                    computer = parse_input_file(input_fpath=computer)
                computer_ = None
                if isinstance(computer, dict) or isinstance(computer, omegaconf.DictConfig):
                    computer_ = ComputerVariable(**computer)
                elif isinstance(computer, ComputerVariable):
                    computer_ = computer
                else:
                    raise RuntimeError(f"Unknown input for computer with a type of {computer}.")
                computers_.append(computer_)
            computers = computers_
        else:
            raise Exception()

        # TODO: We'd better make computer_chain as a class?
        num_workers_per_chain = len(computers[0].value)

        # Add the first chain
        value = [[] for _ in range(num_workers_per_chain)]  # size (num_chainsteps, num_chains)
        for i, computer in enumerate(computers):  # i -> chainstep index
            num = len(computer.value)
            if num != num_workers_per_chain:
                raise Exception(f"chainstep.{i:>02d} need {num_workers_per_chain:>2d} workers but got {num:>2d}.")
            for j, worker in enumerate(computer.value):  # j -> chain index
                value[j].append(worker)

        return value

    def as_dict(self) -> dict:
        """"""

        return self._init_params


@registers.variable.register
class ComputerVariable(Variable):

    def __init__(
        self,
        potter,
        driver={},
        scheduler={},
        *,
        use_grid: bool = False,
        estimate_uncertainty: Optional[bool] = None,
        switch_backend: Optional[str] = None,
        batchsize: int = 1,
        share_wdir: bool = False,
        use_single: bool = False,
        retain_info: bool = False,
        directory = "./",
    ):
        """"""
        # Save input parameters
        self._init_params = copy.deepcopy(
            dict(
                potter=potter,
                driver=driver,
                scheduler=scheduler,
                use_grid=use_grid,
                estimate_uncertainty=estimate_uncertainty,
                switch_backend=switch_backend,
                batchsize=batchsize,
                share_wdir=share_wdir,
                use_single=use_single,
                retain_info=retain_info,
            )
        )

        # Canonicalise components
        self.potter = broadcast_and_adjust_potter(
            potter,
            estimate_uncertainty=estimate_uncertainty,
            switch_backend=switch_backend,
            print_func=self._print,
        )
        self.driver = self._canonicalise_driver(driver)
        self.scheduler = self._canonicalise_scheduler(scheduler)

        # This can be updated in the compute operation.
        self.batchsize = batchsize

        workers = self._broadcast_workers(
            self.potter,
            self.driver,
            self.scheduler,
            use_grid=use_grid,
            batchsize=self.batchsize,
            share_wdir=share_wdir,
            use_single=use_single,
            retain_info=retain_info,
        )
        super().__init__(workers, directory=directory)

        self.use_single = use_single

        return

    def _canonicalise_driver(self, inp) -> list[dict]:
        """Load drivers from a Variable or a dict."""
        # print("driver: ", inp)
        drivers = []  # params
        if isinstance(inp, Variable):
            drivers = inp.value
        elif isinstance(inp, list):  # assume it contains a list of dicts
            drivers = inp
        elif isinstance(inp, dict) or isinstance(
            inp, omegaconf.dictconfig.DictConfig
        ):  # assume it only contains one driver
            driver_params = copy.deepcopy(inp)
            # driver = self.potter.create_driver(driver_params) # use external backend
            drivers = [driver_params]
        else:
            raise RuntimeError(f"Unknown {inp} for drivers.")

        return drivers

    def _canonicalise_scheduler(self, inp):
        """"""
        scheduler = None
        if isinstance(inp, BaseScheduler):
            scheduler = inp
        elif isinstance(inp, Variable):
            scheduler = inp.value
        elif isinstance(inp, dict) or isinstance(inp, omegaconf.DictConfig):
            scheduler_params = copy.deepcopy(inp)
            backend = scheduler_params.pop("backend", "local")
            scheduler = registers.create("scheduler", backend, convert_name=True, **scheduler_params)
        else:
            raise RuntimeError(f"Unknown {inp} for the scheduler.")

        return scheduler

    def _broadcast_workers(
        self,
        potters,
        drivers,
        scheduler,
        *,
        use_grid: bool = False,
        batchsize: int = 1,
        share_wdir: bool = False,
        use_single: bool = False,
        retain_info: bool = False,
    ) -> list[BaseWorker]:
        """Create a list of workers."""
        # check potters
        num_potters = len(potters)
        self._print(f"{num_potters =}")

        # check if there were custom wdirs, and zip longest
        num_drivers = len(drivers)

        # broadcast
        pairs = list(itertools.product(range(num_drivers), range(num_potters)))

        if not use_grid:
            num_pairs = len(pairs)
            wdirs = [self.directory / f"w{i}" for i in range(num_pairs)]

            # create workers
            workers = []
            for i, (d_i, p_i) in enumerate(pairs):
                # workers share calculator in potter
                driver = potters[p_i].create_driver(drivers[d_i])
                if isinstance(driver, BaseDriver):
                    if not use_single:
                        worker = DriverBasedWorker(potters[p_i], driver, scheduler)
                    else:
                        worker = SingleWorker(potters[d_i], driver, scheduler)
                    worker._share_wdir = share_wdir
                    worker._retain_info = retain_info
                elif isinstance(driver, BaseReactor):
                    worker = ReactorBasedWorker(potters[p_i], driver, scheduler)
                else:
                    raise Exception()  # Driver should already be checked by create_driver.
                # wdir is temporary as it may be reset by the compute operation
                worker.directory = wdirs[i]
                workers.append(worker)

            for worker in workers:
                worker.batchsize = batchsize
        else:
            new_potters, new_drivers = [], []
            for d_i, p_i in pairs:
                potter = potters[p_i]
                new_potters.append(potter)
                driver = potter.create_driver(drivers[d_i])
                new_drivers.append(driver)
            worker = GridDriverBasedWorker(new_potters, new_drivers, scheduler=scheduler)
            worker.batchsize = batchsize
            worker.directory = self.directory
            workers = [worker]

        return workers

    def as_dict(self) -> dict:
        """"""

        return self._init_params


if __name__ == "__main__":
    ...
