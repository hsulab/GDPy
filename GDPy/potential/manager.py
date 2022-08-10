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
    potential_names = ["Vasp", "DP", "EANN", "Lasp", "NequIP"]

    def __init__(self):
        """
        """
        # collect registered managers
        self.registered_potentials = {}
        managers = importlib.import_module("GDPy.potential.potential")
        for pot_name in self.potential_names:
            self.registered_potentials[pot_name] = getattr(managers, pot_name+self.SUFFIX)

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


def create_manager(input_json):
    """create a potential manager"""
    # create potential manager
    #with open(input_json, 'r') as fopen:
    #    pot_dict = json.load(fopen)
    pot_dict = parse_input_file(input_json)
    train_dict = pot_dict.get("training", None)
    mpm = PotManager() # main potential manager
    pm = mpm.create_potential(
        pot_dict["name"], train_dict,
        pot_dict["backend"], 
        **pot_dict["kwargs"]
    )
    #print(pm.models)

    return pm

def create_manager_new(input_dict):
    """"""
    atype_map = {}
    for i, a in enumerate(input_dict["calc_params"]["type_list"]):
        atype_map[a] = i

    # create potential
    mpm = PotManager() # main potential manager
    eann_pot = mpm.create_potential(
        pot_name = input_dict["name"],
        # TODO: remove this kwargs
        backend = "ase",
        models = input_dict["calc_params"]["pair_style"]["model"],
        type_map = atype_map
    )

    worker, run_params = eann_pot.create_worker(
        backend = input_dict["backend"],
        calc_params = input_dict["calc_params"],
        dyn_params = input_dict["dyn_params"]
    )
    print(run_params)

    return worker, run_params

def create_pot_manager(input_file=None, calc_name="calc1"):
    """"""
    if input_file is not None:
        pot_dict = parse_input_file(input_file)

        # - find calculators

        mpm = PotManager() # main potential manager
        pm = mpm.create_potential(pot_name = pot_dict["name"])
        pm.register_calculator(pot_dict["calculators"][calc_name])
        pm.version = calc_name
    else:
        pm = None
    
    return pm

if __name__ == "__main__":
    # test old manager
    #pm = PotManager()
    #pot = pm.create_potential('DP', miaow='xx')
    #from GDPy.potential.potential import NPManager
    #pm.register_potential('NP', NPManager)
    #pot = pm.create_potential('NP', what='nani')
    # test new
    pot_dict = dict(
        name = "eann",
        backend = "lammps",
        kwargs  = [] # parameters for the backend
    )
    dyn_dict = {
        # check current pot's backend is valid for this task
        "task": "min",
        "kwargs": {
            "steps": 400,
            "fmax": 0.05,
            "repeat": 1
        }
    }
    pass