#!/usr/bin/env python3
# -*- coding: utf-8 -*

import json
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
    
    def create_potential(self, pot_name, train_dict=None, *args, **kwargs):
        """
        """
        if pot_name in self.potential_names:
            pot_class = self.registered_potentials[pot_name]
            potential = pot_class(*args, **kwargs)
            if train_dict is not None:
                potential.register_training(train_dict)
        else:
            raise NotImplementedError('%s is not registered as a potential.' %(pot_name))

        return potential


def create_potter(config_file=None):
    """"""
    potter = None
    if config_file:
        params = parse_input_file(config_file)
        manager = PotManager()

        # - calculator
        potential_params = params.get("potential", None)
        if not potential_params:
            potential_params = params

        name = potential_params.get("name", None)
        potter = manager.create_potential(pot_name=name)
        potter.register_calculator(potential_params.get("params", {}))
        potter.version = potential_params.get("version", "unknown")

        # - scheduler
        scheduler_params = params.get("scheduler", None)
        if scheduler_params:
            potter.register_scheduler(scheduler_params)

    return potter

if __name__ == "__main__":
    pass