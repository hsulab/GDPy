#!/usr/bin/env python3
# -*- coding: utf-8 -*

import sys
import inspect
import importlib
import typing
import pathlib

from GDPy.utils.command import parse_input_file

from GDPy.core.register import registers
from GDPy.scheduler.interface import create_scheduler
from GDPy.potential.manager import AbstractPotentialManager
TManager = typing.TypeVar("TManager", bound="AbstractPotentialManager")


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
    scheduler = create_scheduler(scheduler_params)

    # - try worker
    if driver and scheduler:
        if scheduler.name == "local":
            from GDPy.computation.worker.drive import CommandDriverBasedWorker as Worker
            run_worker = Worker(potter, driver, scheduler)
        else:
            from GDPy.computation.worker.drive import QueueDriverBasedWorker as Worker
            run_worker = Worker(potter, driver, scheduler)
        print(run_worker)
    
    # - final worker
    worker = (run_worker if not train_worker else train_worker)

    batchsize = params.get("batchsize", 1)
    worker.batchsize = batchsize
    
    return worker

if __name__ == "__main__":
    pass