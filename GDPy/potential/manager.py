#!/usr/bin/env python3
# -*- coding: utf-8 -*

import importlib
import typing

from GDPy.utils.command import parse_input_file

# from GDPy.potential.potential import AbstractPotential
TManager = typing.TypeVar("TManager", bound="AbstractPotential")

class PotManager():

    SUFFIX = "Manager"
    potential_names = ["vasp", "dp", "eann", "lasp", "nequip"]

    def __init__(self):
        """
        """
        # collect registered managers
        self.registered_potentials = {}
        managers = importlib.import_module("GDPy.potential.potential")
        for pot_name in self.potential_names:
            self.registered_potentials[pot_name] = getattr(managers, pot_name.capitalize()+self.SUFFIX)

        return
    
    def register_potential(self, pot_name: str, pot_class: typing.Type[TManager]):
        """
        Register a custom potential manager class
        """
        self.potential_names.append(pot_name)
        self.registered_potentials[pot_name] = pot_class

        return
    
    def create_potential(self, pot_name, train_params=None, *args, **kwargs):
        """
        """
        if pot_name in self.potential_names:
            pot_class = self.registered_potentials[pot_name]
            potential = pot_class(*args, **kwargs)
            if train_params is not None:
                potential.register_training(train_params)
        else:
            raise NotImplementedError('%s is not registered as a potential.' %(pot_name))

        return potential


def create_potter(config_file=None):
    """"""
    params = parse_input_file(config_file)

    potter, train_worker, driver, run_worker = None, None, None, None

    # - get potter first
    potential_params = params.get("potential", {})
    if not potential_params:
        potential_params = params
    manager = PotManager()
    name = potential_params.get("name", None)
    potter = manager.create_potential(pot_name=name)
    potter.register_calculator(potential_params.get("params", {}))
    potter.version = potential_params.get("version", "unknown")

    # - scheduler for training the potential
    train_params = potential_params.get("trainer", {})
    if train_params:
        from GDPy.computation.worker.train import TrainWorker
        potter.register_trainer(train_params)
        train_worker = TrainWorker(potter, potter.train_scheduler)

    # - try to get driver
    driver_params = params.get("driver", {})
    if potter.calc:
        driver = potter.create_driver(driver_params) # use external backend

    # - scheduler for running the potential
    scheduler_params = params.get("scheduler", {})
    if scheduler_params:
        potter.register_scheduler(scheduler_params)

    # - try worker
    if driver and potter.scheduler:
        from GDPy.computation.worker.drive import DriverBasedWorker
        run_worker = DriverBasedWorker(driver, potter.scheduler)
    
    # TODO: cant define train and run at the same time?

    return (potter if not train_worker else train_worker)

if __name__ == "__main__":
    pass