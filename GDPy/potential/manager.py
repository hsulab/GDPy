#!/usr/bin/env python3
# -*- coding: utf-8 -*

"""
Potential Manager
deals with various machine learning potentials
"""

import abc
import copy
from typing import Union, List, NoReturn

import numpy as np

from GDPy.scheduler import create_scheduler


class AbstractPotentialManager(abc.ABC):
    """
    Create various potential instances
    """

    name = "potential"
    version = "m00" # calculator name

    implemented_backends = []
    valid_combinations = []

    _calc = None
    _scheduler = None

    def __init__(self):
        """
        """

        self.uncertainty = None # uncertainty estimation method

        return
    
    @property
    def calc(self):
        return self._calc
    
    @calc.setter
    def calc(self, calc_):
        self._calc = calc_
        return 
    
    @property
    def scheduler(self):
        return self._scheduler
    
    @scheduler.setter
    def scheduler(self, scheduler_):
        self._scheduler = scheduler_
        return
    
    @abc.abstractmethod
    def register_calculator(self, calc_params, *agrs, **kwargs):
        """ register calculator
        """
        self.calc_backend = calc_params.pop("backend", self.name)
        if self.calc_backend not in self.implemented_backends:
            raise RuntimeError(f"Unknown backend for potential {self.name}")

        self.calc_params = copy.deepcopy(calc_params)

        return

    def create_driver(
        self, 
        dyn_params: dict = {},
        *args, **kwargs
    ):
        """ create a worker for dynamics
            default the dynamics backend will be the same as calc
            however, ase-based dynamics can be used for all calculators
        """
        if not hasattr(self, "calc") or self.calc is None:
            raise AttributeError("Cant create driver before a calculator has been properly registered.")
            
        # parse backends
        self.dyn_params = dyn_params
        dynamics = dyn_params.get("backend", self.calc_backend)
        if dynamics == "external":
            dynamics = self.calc_backend

        if [self.calc_backend, dynamics] not in self.valid_combinations:
            raise RuntimeError(f"Invalid dynamics backend {dynamics} based on {self.calc_backend} calculator")
        
        # create dynamics
        calc = self.calc

        if dynamics == "ase":
            from GDPy.computation.ase import AseDriver as driver_cls
        elif dynamics == "lammps":
            from GDPy.computation.lammps import LmpDriver as driver_cls
        elif dynamics == "lasp":
            from GDPy.computation.lasp import LaspDriver as driver_cls
        elif dynamics == "vasp":
            from GDPy.computation.vasp import VaspDriver as driver_cls

        driver = driver_cls(calc, dyn_params, directory=calc.directory)
        driver.pot_params = self.as_dict()
        
        return driver
    
    def register_scheduler(self, params_, *args, **kwargs) -> NoReturn:
        """ register machine used to submit jobs
        """
        params = copy.deepcopy(params_)
        scheduler = create_scheduler(params)

        self.scheduler = scheduler

        return

    def create_worker(self, driver, scheduler, *args, **kwargs):
        """ a machine operates calculations submitted to queue
        """
        # TODO: check driver is consistent with this potential

        from GDPy.computation.worker.worker import DriverBasedWorker
        worker = DriverBasedWorker(driver, scheduler)

        # TODO: check if scheduler resource satisfies driver's command

        return worker

    def register_trainer(self, train_params_: dict):
        """"""
        train_params = copy.deepcopy(train_params_)
        self.train_config = train_params.get("config", None)

        self.train_size = train_params.get("size", 1)
        self.train_dataset = train_params.get("dataset", None)

        # - training
        self.train_epochs = train_params.get("epochs", 500)

        # - for the train-valid split
        self.train_split_ratio = train_params.get("split_ratio", 0.9)
        # TODO: manager also needs a rng to handle random?
        self.train_split_seed = train_params.get("split_seed", np.random.randint(0,10000))

        self.train_batchsize = self.get_batchsize(self.train_config)

        # - for the task
        scheduelr_params = train_params.get("scheduler", {}) 
        self.train_scheduler = create_scheduler(scheduelr_params)

        train_command = train_params.get("train", None)
        self.train_command = train_command

        freeze_command = train_params.get("freeze", None)
        self.freeze_command = freeze_command

        return
    
    def get_batchsize(self, *args, **kwargs):
        """ get batchsize for train and valid set
        """

        return
    
    def as_dict(self):
        """"""
        params = {}
        params["name"] = self.name

        pot_params = {"backend": self.calc_backend}
        pot_params.update(copy.deepcopy(self.calc_params))
        params["params"] = pot_params

        # TODO: add train params?

        return params


if __name__ == "__main__":
    pass