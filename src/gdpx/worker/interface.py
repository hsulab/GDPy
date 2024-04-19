#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
import itertools
import pathlib
from typing import NoReturn, Optional, List, Callable

import omegaconf

from ..core.operation import Operation
from ..core.variable import Variable
from ..core.register import registers
from ..utils.command import parse_input_file

from ..potential.manager import AbstractPotentialManager
from ..scheduler.scheduler import AbstractScheduler
from .worker import AbstractWorker
from .drive import DriverBasedWorker, CommandDriverBasedWorker, QueueDriverBasedWorker
from .react import ReactorBasedWorker
from .single import SingleWorker


def convert_config_to_potter(config):
    """Convert a configuration file or a dict to a potter/reactor.

    This function is only called in the `main.py`.

    """
    if isinstance(config, dict):
        params = config
    else:  # assume it is json or yaml
        params = parse_input_file(input_fpath=config)

    ptype = params.pop("type", "computer")

    # NOTE: compatibility
    potter_params = params.pop("potter", None)
    potential_params = params.pop("potential", None)
    if potter_params is None:
        if potential_params is not None:
            params["potter"] = potential_params
        else:
            raise RuntimeError("Fail to find any potter (potential) definition.")
    else:
        params["potter"] = potter_params

    if ptype == "computer":
        potter = ComputerVariable(**params).value[0]
    elif ptype == "reactor":
        from gdpx.reactor.interface import ReactorVariable

        potter = ReactorVariable(
            potter=params["potter"],
            driver=params.get("driver", None),
            scheduler=params.get("scheduler", {}),
            batchsize=params.get("batchsize", 1),
        ).value[0]
    else:
        ...

    return potter


def convert_input_to_potter(
    inp,
    estimate_uncertainty: Optional[bool] = False,
    switch_backend: Optional[str] = None,
    print_func: Callable = print,
) -> AbstractPotentialManager:
    """Convert an input to a potter and adjust its behaviour."""
    potter = None
    if isinstance(inp, AbstractPotentialManager):
        potter = inp
    elif isinstance(inp, Variable):
        potter = inp.value
    elif isinstance(inp, dict) or isinstance(inp, omegaconf.dictconfig.DictConfig):
        potter_params = copy.deepcopy(inp)
        name = potter_params.get("name", None)
        potter = registers.create(
            "manager",
            name,
            convert_name=True,
        )
        potter.register_calculator(potter_params.get("params", {}))
        potter.version = potter_params.get("version", "unknown")
    elif isinstance(inp, str) or isinstance(inp, pathlib.Path):
        if pathlib.Path(inp).exists():
            potter_params = parse_input_file(input_fpath=inp)
            name = potter_params.get("name", None)
            potter = registers.create(
                "manager",
                name,
                convert_name=True,
            )
            potter.register_calculator(potter_params.get("params", {}))
            potter.version = potter_params.get("version", "unknown")
        else:
            raise RuntimeError(f"The potter configuration `{inp}` does not exist.")
    else:
        raise RuntimeError(f"Unknown {inp} of type {type(inp)} for the potter.")

    # adjust potter behaviour
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

    return potter


@registers.variable.register
class ComputerVariable(Variable):

    def __init__(
        self,
        potter,
        driver={},
        scheduler={},
        *,
        estimate_uncertainty: Optional[bool] = None,
        switch_backend: Optional[str] = None,
        batchsize: int = 1,
        share_wdir: bool = False,
        use_single: bool = False,
        retain_info: bool = False,
        custom_wdirs=None,
        directory=pathlib.Path.cwd(),
    ):
        """"""
        self.potter = convert_input_to_potter(
            potter,
            estimate_uncertainty=estimate_uncertainty,
            switch_backend=switch_backend,
            print_func=self._print,
        )
        self.driver = self._load_driver(driver)
        self.scheduler = self._load_scheduler(scheduler)

        # - ...
        self.batchsize = batchsize  # NOTE: This can be updated in drive operation.

        workers = self._create_workers(
            self.potter,
            self.driver,
            self.scheduler,
            self.batchsize,
            share_wdir=share_wdir,
            use_single=use_single,
            retain_info=retain_info,
            custom_wdirs=custom_wdirs,
        )
        super().__init__(workers)

        self.custom_wdirs = None
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

    def _create_workers(
        self,
        potter,
        drivers,
        scheduler,
        batchsize: int = 1,
        share_wdir: bool = False,
        use_single: bool = False,
        retain_info: bool = False,
        custom_wdirs=None,
    ):
        """Create a list of workers."""
        # - check if there were custom wdirs, and zip longest
        ndrivers = len(drivers)
        if custom_wdirs is not None:
            wdirs = [pathlib.Path(p) for p in custom_wdirs]
        else:
            wdirs = [self.directory / f"w{i}" for i in range(ndrivers)]

        nwdirs = len(wdirs)
        assert (nwdirs == ndrivers and ndrivers > 1) or (
            nwdirs >= 1 and ndrivers == 1
        ), "Invalid wdirs and drivers."
        pairs = itertools.zip_longest(wdirs, drivers, fillvalue=drivers[0])

        # - create workers
        # TODO: broadcast potters, schedulers as well?
        workers = []
        for wdir, driver_params in pairs:
            # workers share calculator in potter
            driver = potter.create_driver(driver_params)
            if not use_single:
                if scheduler.name == "local":
                    worker = CommandDriverBasedWorker(potter, driver, scheduler)
                else:
                    worker = QueueDriverBasedWorker(potter, driver, scheduler)
            else:
                worker = SingleWorker(potter, driver, scheduler)
            worker._share_wdir = share_wdir
            worker._retain_info = retain_info
            # wdir is temporary as it may be reset by drive operation
            worker.directory = wdir
            workers.append(worker)

        for worker in workers:
            worker.batchsize = batchsize

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
        self.potter = convert_input_to_potter(
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
            self.potter, self.driver, self.scheduler, self.batchsize
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
        potter,
        drivers: List[dict],
        scheduler,
        batchsize: int = 1,
    ):
        """"""
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
