"""Construct computation workers without depending on workflow Variables."""

from __future__ import annotations

import copy
import itertools
import pathlib
from collections.abc import Mapping, Sequence
from typing import Callable

from ase.calculators.calculator import BaseCalculator

from gdpx.computation.driver import BaseDriver
from gdpx.factory.scheduler import canonicalise_scheduler
from gdpx.potential.manager import BasePotentialManager
from gdpx.potential.utils import convert_input_to_potter
from gdpx.reactor.reactor import BaseReactor
from gdpx.worker.drive import DriverBasedWorker
from gdpx.worker.grid import GridDriverBasedWorker
from gdpx.worker.react import ReactorBasedWorker
from gdpx.worker.single import SingleWorker
from gdpx.worker.worker import BaseWorker


def broadcast_and_adjust_potter(
    inp,
    estimate_uncertainty: bool | None = False,
    switch_backend: str | None = None,
    print_func: Callable = print,
) -> list[BasePotentialManager]:
    potter = convert_input_to_potter(inp)
    if not isinstance(potter, BasePotentialManager):
        raise TypeError(f"Expected BasePotentialManager, got {type(potter).__name__}.")
    potters = potter.broadcast(potter) if hasattr(potter, "broadcast") else [potter]
    if not all(isinstance(item.calc, BaseCalculator) for item in potters):
        raise TypeError("Every potential manager must contain an ASE calculator.")
    for index, item in enumerate(potters):
        print_func(f"potter-{index} {item.name}")
        if estimate_uncertainty is not None and hasattr(item, "switch_uncertainty_estimation"):
            item.switch_uncertainty_estimation(estimate_uncertainty)
        if switch_backend is not None and hasattr(item, "switch_backend"):
            item.switch_backend(backend=switch_backend)
    return potters


def create_workers(config=None, *, directory="./", print_func: Callable = print, **overrides) -> list[BaseWorker]:
    """Create broadcast computation workers from configuration."""
    if isinstance(config, BaseWorker):
        if overrides:
            raise TypeError("Overrides cannot be applied to an existing worker.")
        return [config]
    if isinstance(config, Sequence) and not isinstance(config, (str, bytes, Mapping)):
        if all(isinstance(item, BaseWorker) for item in config):
            return list(config)
    if config is None:
        params = {}
    elif isinstance(config, Mapping):
        params = copy.deepcopy(dict(config))
    else:
        raise TypeError(f"Computer must be a mapping or worker, got {type(config).__name__}.")
    params.update(copy.deepcopy(overrides))
    if "potter" not in params and "potential" in params:
        params["potter"] = params.pop("potential")
    if "potter" not in params:
        raise ValueError("Computer configuration requires `potter` (or legacy `potential`).")

    potters = broadcast_and_adjust_potter(
        params.pop("potter"),
        estimate_uncertainty=params.pop("estimate_uncertainty", None),
        switch_backend=params.pop("switch_backend", None),
        print_func=print_func,
    )
    driver_input = params.pop("driver", {})
    drivers = copy.deepcopy(list(driver_input) if isinstance(driver_input, list) else [dict(driver_input)])
    scheduler = canonicalise_scheduler(params.pop("scheduler", {}))
    use_grid = params.pop("use_grid", False)
    batchsize = params.pop("batchsize", 1)
    share_wdir = params.pop("share_wdir", False)
    use_single = params.pop("use_single", False)
    retain_info = params.pop("retain_info", False)
    if params:
        raise TypeError(f"Unknown computer options: {', '.join(sorted(params))}")

    directory = pathlib.Path(directory)
    pairs = list(itertools.product(range(len(drivers)), range(len(potters))))
    if use_grid:
        grid_potters = [potters[p_index] for _, p_index in pairs]
        grid_drivers = [potters[p_index].create_driver(drivers[d_index]) for d_index, p_index in pairs]
        worker = GridDriverBasedWorker(grid_potters, grid_drivers, scheduler=scheduler)
        worker.batchsize = batchsize
        worker.directory = directory
        return [worker]

    workers: list[BaseWorker] = []
    for index, (driver_index, potter_index) in enumerate(pairs):
        potter = potters[potter_index]
        driver = potter.create_driver(drivers[driver_index])
        if isinstance(driver, BaseDriver):
            worker = SingleWorker(potter, driver, scheduler) if use_single else DriverBasedWorker(potter, driver, scheduler)
            worker._share_wdir = share_wdir
            worker._retain_info = retain_info
        elif isinstance(driver, BaseReactor):
            worker = ReactorBasedWorker(potter, driver, scheduler)
        else:
            raise TypeError(f"Unsupported driver {type(driver).__name__}.")
        worker.batchsize = batchsize
        worker.directory = directory / f"w{index}"
        workers.append(worker)
    return workers


def create_worker_chains(configs, *, directory="./", print_func: Callable = print) -> list[list[BaseWorker]]:
    """Create worker-major chains from a sequence of computer configs."""
    steps = [create_workers(config, directory=directory, print_func=print_func) for config in configs]
    if not steps:
        raise ValueError("A worker chain requires at least one computer configuration.")
    width = len(steps[0])
    if any(len(step) != width for step in steps):
        raise ValueError("Every worker-chain step must create the same number of workers.")
    return [[step[index] for step in steps] for index in range(width)]


def canonicalise_worker(inp_worker):
    """Compatibility adapter returning the first canonical worker."""
    if inp_worker is None:
        return None
    workers = create_workers(inp_worker)
    return workers[0]
