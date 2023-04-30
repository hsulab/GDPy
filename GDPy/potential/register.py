#!/usr/bin/env python3
# -*- coding: utf-8 -*

import sys
import inspect
import importlib
import typing
import pathlib

from GDPy.utils.command import parse_input_file

from GDPy.core.register import registers
from GDPy.potential.manager import AbstractPotentialManager
TManager = typing.TypeVar("TManager", bound="AbstractPotentialManager")

class PotentialRegister():

    def __init__(self):
        """
        """
        self._name_cls_map = dict()

        # collect registered managers
        self.registered_potentials = {}
        managers = importlib.import_module("GDPy.potential.managers")
        for name, data in inspect.getmembers(managers, inspect.ismodule):
            if name.startswith("__"):
                continue
            cls_name = name.capitalize() + "Manager"
            pot_cls = getattr(data, cls_name)
            if pot_cls:
                assert issubclass(pot_cls, AbstractPotentialManager), f"{cls_name} is not a PotentialManager object."
                self._name_cls_map[name] = pot_cls

        return
    
    def register_potential(self, pot_cls: typing.Type[TManager]=None, pot_name: str=None):
        """
        Register a custom potential manager class
        """
        if pot_name is None:
            pot_name = pot_cls.__class__.__name__
        if pot_cls != None:
            self._name_cls_map[pot_name] = pot_cls
        else:
            def wrapper(obj):
                self._name_method_map[pot_name] = obj
                return obj
            return wrapper

        return
    
    def create_potential(self, pot_name, train_params=None, *args, **kwargs) -> AbstractPotentialManager:
        """
        """
        pot_cls = None
        if pot_name in self._name_cls_map:
            pot_cls = self._name_cls_map[pot_name]
        else:
            # - read from input
            if pot_name.endswith(".py"):
                #manager = inspect.getsourcefile(pot_name)
                pot_def_path = pathlib.Path(pot_name).resolve()
                sys.path.append(str(pot_def_path.parent))
                pot_name = pot_def_path.name[:-3]
                manager = importlib.import_module(pot_name)
                pot_cls = getattr(manager, pot_name.capitalize()+"Manager")
                # NOTE: for as_dict, name should be absolute path
                pot_cls.name = str(pot_def_path)

        if pot_cls:
            potential = pot_cls(*args, **kwargs)
            if train_params is not None:
                potential.register_training(train_params)
        else:
            raise NotImplementedError("%s is not registered as a potential." %(pot_name))

        return potential


def create_potter(config_file=None):
    """"""
    params = parse_input_file(config_file)

    potter, train_worker, driver, run_worker = None, None, None, None

    # - get potter first
    potential_params = params.get("potential", {})
    if not potential_params:
        potential_params = params

    # --- specific potential
    name = potential_params.get("name", None)
    #manager = PotentialRegister()
    #potter = manager.create_potential(pot_name=name)
    #potter.register_calculator(potential_params.get("params", {}))
    #potter.version = potential_params.get("version", "unknown")
    potter = registers.create(
        "manager", name, convert_name=True,
    )
    potter.register_calculator(potential_params.get("params", {}))
    potter.version = potential_params.get("version", "unknown")
    print(potter.calc)

    # --- uncertainty estimator
    est_params = potential_params.get("uncertainty", None)
    est_register = getattr(potter, "register_uncertainty_estimator", None)
    if est_params and est_register:
        #print("create estimator!!!!")
        potter.register_uncertainty_estimator(est_params)

    # - scheduler for training the potential
    train_params = potential_params.get("trainer", {})
    if train_params:
        from GDPy.computation.worker.train import TrainWorker
        potter.register_trainer(train_params)
        train_worker = TrainWorker(potter, potter.train_scheduler)

    # - try to get driver
    driver_params = params.get("driver", {})
    if potter.calc:
        print("driver: ", driver)
        driver = potter.create_driver(driver_params) # use external backend
    print("driver: ", driver)

    # - scheduler for running the potential
    scheduler_params = params.get("scheduler", {})
    # default is local machine
    potter.register_scheduler(scheduler_params)

    # - try worker
    if driver and potter.scheduler:
        if potter.scheduler.name == "local":
            from GDPy.computation.worker.drive import CommandDriverBasedWorker as Worker
            run_worker = Worker(potter, driver, potter.scheduler)
        else:
            from GDPy.computation.worker.drive import QueueDriverBasedWorker as Worker
            run_worker = Worker(potter, driver, potter.scheduler)
        print(run_worker)
    
    # - final worker
    worker = (run_worker if not train_worker else train_worker)

    batchsize = params.get("batchsize", 1)
    worker.batchsize = batchsize
    
    return worker

if __name__ == "__main__":
    pass