#!/usr/bin/env python3
# -*- coding: utf-8 -*

import json
import importlib
import typing

# from GDPy.potential.potential import AbstractPotential
TManager = typing.TypeVar("TManager", bound="AbstractPotential")

class PotManager():

    SUFFIX = "Manager"
    potential_names = ["DP", "EANN", "Lasp"]

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
    
    def create_workder(self):
        """
        # TODO: create a worker for single-point or dynamics calculations
        """
        potential = self.calc_dict["potential"]
        if potential == "lasp":
            # lasp has different backends (ase, lammps, lasp)
            from GDPy.calculator.lasp import LaspNN
            self.calc = LaspNN(**self.calc_dict["kwargs"])
        elif potential == "eann": # and inteface to lammps
            # eann has different backends (ase, lammps)
            from GDPy.calculator.lammps import Lammps
            self.calc = Lammps(**self.calc_dict["kwargs"])
        # DFT methods
        elif potential == "vasp":
            from GDPy.calculator.vasp import VaspMachine
            with open(self.calc_dict["kwargs"], "r") as fopen:
                input_dict = json.load(fopen)
            self.calc = VaspMachine(**input_dict)
        else:
            raise ValueError("Unknown potential to calculation...")
        
        interface = self.calc_dict["interface"]
        if interface == "ase":
            from GDPy.calculator.ase_interface import AseDynamics
            self.worker = AseDynamics(self.calc, directory=self.calc.directory)
            # use ase no need to recaclc constraint since atoms has one
            self.cons_indices = None
        else: 
            # find z-axis constraint
            self.cons_indices = None
            if self.system_type == "surface":
                constraint = self.ga_dict["system"]["substrate"]["constraint"]
                if constraint is not None:
                    index_group = constraint.split()
                    indices = []
                    for s in index_group:
                        r = [int(x) for x in s.split(":")]
                        indices.append([r[0]+1, r[1]]) # starts from 1
                self.cons_indices = ""
                for s, e in indices:
                    self.cons_indices += "{}:{} ".format(s, e)
                print("constraint indices: ", self.cons_indices)
        
            if interface == "queue":
                from GDPy.calculator.vasp import VaspQueue
                self.worker = VaspQueue(
                    self.da,
                    tmp_folder = self.CALC_DIRNAME,
                    vasp_machine = self.calc, # vasp machine
                    n_simul = self.calc_dict["nparallel"], 
                    prefix = self.calc_dict["prefix"],
                    repeat = self.calc_dict["repeat"] # TODO: add this to minimsation with fmax and steps
                )
            elif interface == "lammps":
                from GDPy.calculator.lammps import LmpDynamics as dyn
                # use lammps optimisation
                self.worker = dyn(
                    self.calc, directory=self.calc.directory
                )
            elif interface == "lasp":
                from GDPy.calculator.lasp import LaspDynamics as dyn
                self.worker = dyn(
                    self.calc, directory=self.calc.directory
                )
            else:
                raise ValueError("Unknown interface to optimisation...")

        return

def create_manager(input_json):
    """create a potential manager"""
    # create potential manager
    with open(input_json, 'r') as fopen:
        pot_dict = json.load(fopen)
    train_dict = pot_dict.get("training", None)
    mpm = PotManager() # main potential manager
    pm = mpm.create_potential(
        pot_dict["name"], train_dict,
        pot_dict["backend"], 
        **pot_dict["kwargs"]
    )
    #print(pm.models)

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