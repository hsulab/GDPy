#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
import itertools
import logging
import pathlib
from typing import NoReturn, Optional, List

import numpy as np

import omegaconf

from ase import Atoms
from ase.io import read, write
from ase.geometry import find_mic

from ..core.operation import Operation
from ..core.variable import Variable
from ..core.register import registers
from ..utils.command import parse_input_file

from ..computation.driver import AbstractDriver
from ..data.array import AtomsNDArray
from ..potential.manager import AbstractPotentialManager
from ..reactor.reactor import AbstractReactor
from ..scheduler.scheduler import AbstractScheduler
from .worker import AbstractWorker
from .drive import DriverBasedWorker, CommandDriverBasedWorker, QueueDriverBasedWorker
from .single import SingleWorker

DEFAULT_MAIN_DIRNAME = "MyWorker"


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


def convert_input_to_potter(inp):
    """"""
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
        # - Adjust potter calculator behaviour here...
        self.potter = convert_input_to_potter(potter)

        if hasattr(self.potter, "switch_uncertainty_estimation"):
            if estimate_uncertainty is not None:
                self._print(
                    f"{self.potter.name} switches its uncertainty estimation to {estimate_uncertainty}..."
                )
                self.potter.switch_uncertainty_estimation(estimate_uncertainty)
            else:
                ...
        else:
            self._print(
                f"{self.potter.name} does not support switching its uncertainty estimation..."
            )

        if hasattr(self.potter, "switch_backend"):
            if switch_backend is not None:
                self._print(
                    f"{self.potter.name} switches its backend to {switch_backend}..."
                )
                self.potter.switch_backend(backend=switch_backend)
            else:
                ...
        else:
            self._print(f"{self.potter.name} does not support switching its backend...")

        # - ...
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

    def _update_workers(self, potter_node):
        """"""
        if isinstance(potter_node, Variable):
            potter = potter_node.value
        elif isinstance(potter_node, Operation):
            # TODO: ...
            node = potter_node
            if node.preward():
                node.inputs = [input_node.output for input_node in node.input_nodes]
                node.output = node.forward(*node.inputs)
            else:
                print("wait previous nodes to finish...")
            potter = node.output
        else:
            ...
        print("update manager: ", potter)
        print(potter.calc.model_path)
        workers = self._create_workers(
            potter, self.driver, self.scheduler, custom_wdirs=self.custom_wdirs
        )
        self.value = workers

        return

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


def run_worker(
    structure: List[str],
    directory=pathlib.Path.cwd() / DEFAULT_MAIN_DIRNAME,
    worker: DriverBasedWorker = None,
    output: str = None,
    batch: int = None,
    spawn: bool = False,
    archive: bool = False,
):
    """"""
    # - some imported packages change `logging.basicConfig`
    #   and accidently add a StreamHandler to logging.root
    #   so remove it...
    for h in logging.root.handlers:
        if isinstance(h, logging.StreamHandler) and not isinstance(
            h, logging.FileHandler
        ):
            logging.root.removeHandler(h)

    directory = pathlib.Path(directory)
    if not directory.exists():
        directory.mkdir()

    # - read structures
    from gdpx.builder import create_builder

    frames = []
    for i, s in enumerate(structure):
        builder = create_builder(s)
        builder.directory = directory / "init" / f"s{i}"
        frames.extend(builder.run())

    # - find input frames
    worker.directory = directory

    _ = worker.run(frames, batch=batch)
    worker.inspect(resubmit=True)
    if not spawn and worker.get_number_of_running_jobs() == 0:
        # BUG: bacthes may conflict to save results
        # - report
        res_dir = directory / "results"
        if not res_dir.exists():
            res_dir.mkdir(exist_ok=True)

            ret = worker.retrieve(include_retrieved=True, use_archive=archive)
            if not isinstance(worker.driver, AbstractReactor):
                end_frames = [traj[-1] for traj in ret]
                write(res_dir / "end_frames.xyz", end_frames)
            else:
                ...

            # AtomsNDArray(ret).save_file(res_dir/"trajs.h5")
        else:
            print("Results have already been retrieved.")

    return


if __name__ == "__main__":
    ...
