#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
import itertools
import pathlib
from typing import Callable, List, NoReturn, Optional

import omegaconf
from ase.calculators.calculator import Calculator, BaseCalculator

from ..core.operation import Operation
from ..core.register import registers
from ..core.variable import Variable
from ..potential.manager import AbstractPotentialManager
from ..potential.utils import convert_input_to_potter
from ..scheduler.scheduler import AbstractScheduler
from ..utils.command import parse_input_file
from .drive import (CommandDriverBasedWorker, DriverBasedWorker,
                    QueueDriverBasedWorker)
from .grid import GridDriverBasedWorker
from .react import ReactorBasedWorker
from .single import SingleWorker
from .worker import AbstractWorker


def broadcast_and_adjust_potter(
    inp,
    estimate_uncertainty: Optional[bool] = False,
    switch_backend: Optional[str] = None,
    print_func: Callable = print,
) -> List[AbstractPotentialManager]:
    """Convert an input to a potter and adjust its behaviour."""
    # convert everything into potter
    potter = convert_input_to_potter(inp)

    # HACK: broadcast potters
    if hasattr(potter, "broadcast"):
        potters = potter.broadcast(potter)
    else:
        potters = [potter]

    for p in potters:
        assert isinstance(p.calc, BaseCalculator), f"{p.calc} is not `Calculator`."

    # adjust potter behaviour
    for i, potter in enumerate(potters):
        print_func(f"potter-{i} {potter.name}")
        if hasattr(potter, "switch_uncertainty_estimation"):
            if estimate_uncertainty is not None:
                print_func(
                    f"{potter.name} switches its uncertainty estimation to {estimate_uncertainty}..."
                )
                potter.switch_uncertainty_estimation(estimate_uncertainty)
            else:
                ...
        else:
            print_func(
                f"{potter.name} does not support switching its uncertainty estimation..."
            )

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
        directory=pathlib.Path.cwd(),
    ):
        """"""
        self.potter = broadcast_and_adjust_potter(
            potter,
            estimate_uncertainty=estimate_uncertainty,
            switch_backend=switch_backend,
            print_func=self._print,
        )
        self.driver = self._load_driver(driver)
        self.scheduler = self._load_scheduler(scheduler)

        # - ...
        self.batchsize = batchsize  # NOTE: This can be updated in drive operation.

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
        super().__init__(workers)

        self.use_single = use_single

        return

    def _load_driver(self, inp) -> List[dict]:
        """Load drivers from a Variable or a dict."""
        # print("driver: ", inp)
        drivers = []  # params
        if isinstance(inp, Variable):
            drivers = inp.value
        elif isinstance(inp, list):  # assume it contains a List of dicts
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

    def _load_scheduler(self, inp):
        """"""
        scheduler = None
        if isinstance(inp, AbstractScheduler):
            scheduler = inp
        elif isinstance(inp, Variable):
            scheduler = inp.value
        elif isinstance(inp, dict):
            scheduler_params = copy.deepcopy(inp)
            backend = scheduler_params.pop("backend", "local")
            scheduler = registers.create(
                "scheduler", backend, convert_name=True, **scheduler_params
            )
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
    ) -> List[AbstractWorker]:
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
            # for i, (p_i, d_i) in enumerate(pairs):
            for i, (d_i, p_i) in enumerate(pairs):
                # workers share calculator in potter
                driver = potters[p_i].create_driver(drivers[d_i])
                if not use_single:
                    if scheduler.name == "local":
                        worker = CommandDriverBasedWorker(
                            potters[p_i], driver, scheduler
                        )
                    else:
                        worker = QueueDriverBasedWorker(potters[p_i], driver, scheduler)
                else:
                    worker = SingleWorker(potters[d_i], driver, scheduler)
                worker._share_wdir = share_wdir
                worker._retain_info = retain_info
                # wdir is temporary as it may be reset by drive operation
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
            worker = GridDriverBasedWorker(
                new_potters, new_drivers, scheduler=scheduler
            )
            worker.batchsize = batchsize
            worker.directory = self.directory
            workers = [worker]

        return workers


@registers.variable.register
class ReactorVariable(Variable):
    """Create a ReactorBasedWorker.

    TODO:
        Broadcast driver params to give several workers?

    """

    def __init__(
        self,
        potter,
        driver: dict,
        scheduler={},
        *,
        estimate_uncertainty: Optional[bool] = None,
        switch_backend: Optional[str] = None,
        batchsize=1,
        directory="./",
    ):
        """"""
        # - save state by all nodes
        self.potter = broadcast_and_adjust_potter(
            potter,
            estimate_uncertainty=estimate_uncertainty,
            switch_backend=switch_backend,
            print_func=self._print,
        )
        self.driver = self._load_driver(driver)
        self.scheduler = self._load_scheduler(scheduler)

        self.batchsize = batchsize

        # - create a reactor
        # reactor = self.potter.create_reactor(kwargs)
        workers = self._create_workers(
            self.potter, self.driver, self.scheduler, batchsize=self.batchsize
        )

        super().__init__(initial_value=workers, directory=directory)

        return

    def _load_driver(self, inp) -> List[dict]:
        """Load drivers from a Variable or a dict."""
        # print("driver: ", inp)
        drivers = []  # params
        if isinstance(inp, Variable):
            drivers = inp.value
        elif isinstance(inp, list):  # assume it contains a List of dicts
            drivers = inp
        elif isinstance(inp, dict) or isinstance(inp, omegaconf.dictconfig.DictConfig):
            driver_params = copy.deepcopy(inp)
            # driver = self.potter.create_driver(driver_params) # use external backend
            drivers = [driver_params]
        else:
            raise RuntimeError(f"Unknown {inp} for drivers.")

        return drivers

    def _load_scheduler(self, inp):
        """"""
        scheduler = None
        if isinstance(inp, Variable):
            scheduler = inp.value
        elif isinstance(inp, dict):
            scheduler_params = copy.deepcopy(inp)
            backend = scheduler_params.pop("backend", "local")
            scheduler = registers.create(
                "scheduler", backend, convert_name=True, **scheduler_params
            )
        else:
            raise RuntimeError(f"Unknown {inp} for the scheduler.")

        return scheduler

    def _create_workers(
        self,
        potters,
        drivers: List[dict],
        scheduler,
        *,
        batchsize: int = 1,
    ):
        """"""
        # FIXME: Support several potters?
        num_potters = len(potters)
        assert num_potters == 1, "Reactor only supports one potter."
        potter = potters[0]

        workers = []
        for driver_params in drivers:
            driver = potter.create_reactor(driver_params)
            worker = ReactorBasedWorker(potter, driver, scheduler)
            workers.append(worker)

        for worker in workers:
            worker.batchsize = batchsize

        return workers


if __name__ == "__main__":
    ...
