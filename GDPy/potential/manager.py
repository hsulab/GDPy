#!/usr/bin/env python3
# -*- coding: utf-8 -*

import abc
import copy
from typing import Union, List, NoReturn

import numpy as np

from ase.calculators.calculator import Calculator, all_properties, all_changes

"""The abstract base class of any potential manager.

"""

class DummyCalculator(Calculator):

    name = "dummy"

    def __init__(self, restart=None, label="dummy", atoms=None, directory=".", **kwargs):
        super().__init__(restart, label=label, atoms=atoms, directory=directory, **kwargs)

        return

    def calculate(self, atoms=None, properties=all_properties, system_changes=all_changes):
        """"""
        raise NotImplementedError("DummyCalculator is unable to calculate.")


class AbstractPotentialManager(abc.ABC):
    """
    Create various potential instances
    """

    name = "potential"
    version = "m00" # calculator name

    implemented_backends = []
    valid_combinations = []

    _calc = None
    modifier = None

    _estimator = None

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
    
    @abc.abstractmethod
    def register_calculator(self, calc_params, *agrs, **kwargs):
        """Register the host calculator.
        """
        self.calc_backend = calc_params.pop("backend", self.name)
        if self.calc_backend not in self.implemented_backends:
            raise RuntimeError(f"Unknown backend {self.calc_backend} for potential {self.name}")

        self.calc_params = copy.deepcopy(calc_params)

        # - check if there were any modifiers
        modifier_params = calc_params.pop("modifier", None)
        if modifier_params is not None:
            self.register_modifier(modifier_params)

        return
    
    def register_modifier(self, params, *args, **kwargs):
        """Register a modifier.
       
        The modifier is also a calculator but it usually works as an add-on to slightly
        change the host calculator. The modifier will be mixed with the host driver when
        creating the driver by using ase Mixer or driver built-in.

        """
        params = copy.deepcopy(params)
        backend = params.pop("backend", "plumed")
        if backend == "plumed":
            ...
        elif backend == "afir":
            from GDPy.computation.bias.afir import AFIRCalculator
            modifier = AFIRCalculator(**params)
        else:
            ...
        self.modifier = modifier

        return

    def register_uncertainty_estimator(self, est_params_: dict):
        """Create an extra uncertainty estimator.

        This can be used when the current calculator is not capable of 
        estimating uncertainty.
        
        """
        from GDPy.computation.uncertainty import create_estimator
        self._estimator = create_estimator(est_params_, self.calc_params, self._create_calculator)

        return

    def create_driver(
        self, 
        dyn_params: dict = {},
        *args, **kwargs
    ):
        """Create a driver for dynamics.

        Default the dynamics backend will be the same as calc. However, 
        ase-based dynamics can be used for all calculators.

        """
        # - check whether there is a calc
        #if not hasattr(self, "calc") or self.calc is None:
        #    raise AttributeError("Cant create driver before a calculator has been properly registered.")
        if not hasattr(self, "calc"):
            raise AttributeError("Cant create driver before a calculator has been properly registered.")
            
        # parse backends
        self.dyn_params = dyn_params
        dynamics = dyn_params.get("backend", self.calc_backend)
        if dynamics == "external":
            dynamics = self.calc_backend

        if [self.calc_backend, dynamics] not in self.valid_combinations:
            raise RuntimeError(f"Invalid dynamics backend {dynamics} based on {self.calc_backend} calculator")
        
        # - merge params for compat
        merged_params = {}
        if "task" in dyn_params:
            merged_params.update(task=dyn_params.get("task", "min"))

        if "init" in dyn_params or "run" in dyn_params:
            merged_params.update(**dyn_params.get("init", {}))
            merged_params.update(**dyn_params.get("run", {}))
        else:
            merged_params.update(**dyn_params)

        # - other params
        ignore_convergence = merged_params.pop("ignore_convergence", False)

        # - check bias params
        bias_params = self.dyn_params.get("bias", None)
        
        # create dynamics
        calc = self.calc

        if dynamics == "ase":
            from GDPy.computation.asedriver import AseDriver as driver_cls
        elif dynamics == "lammps":
            from GDPy.computation.lammps import LmpDriver as driver_cls
        elif dynamics == "lasp":
            from GDPy.computation.lasp import LaspDriver as driver_cls
        elif dynamics == "vasp":
            from GDPy.computation.vasp import VaspDriver as driver_cls

        # -- add PES modifier (BIAS) to the host PES
        if self.modifier is not None:
            from GDPy.computation.mixer import AddonCalculator
            calc = AddonCalculator(self.calc, self.modifier, 1., 1.)

        driver = driver_cls(
            calc, merged_params, directory=calc.directory, 
            ignore_convergence=ignore_convergence
        )
        driver.pot_params = self.as_dict()
        
        return driver

    def as_dict(self):
        """"""
        params = {}
        params["name"] = self.name

        pot_params = {"backend": self.calc_backend}
        pot_params.update(copy.deepcopy(self.calc_params))
        params["params"] = pot_params

        return params


if __name__ == "__main__":
    pass