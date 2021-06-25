#!/usr/bin/env python3
# -*- coding: utf-8 -*

import importlib
import typing

# from GDPy.potential.potential import AbstractPotential
TManager = typing.TypeVar("TManager", bound="AbstractPotential")

class PotManager():

    SUFFIX = 'Manager'
    potential_names = ['DP']

    def __init__(self):
        """
        """
        # collect registered managers
        self.registered_potentials = {}
        managers = importlib.import_module('GDPy.potential.potential')
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
    
    def create_potential(self, pot_name, *args, **kwargs):
        """
        """
        if pot_name in self.potential_names:
            pot_class = self.registered_potentials[pot_name]
            potential = pot_class(*args, **kwargs)
        else:
            raise NotImplementedError('%s is not registered as a potential.' %(pot_name))

        return potential


if __name__ == '__main__':
    pm = PotManager()
    pot = pm.create_potential('DP', miaow='xx')
    from GDPy.potential.potential import NPManager
    pm.register_potential('NP', NPManager)
    pot = pm.create_potential('NP', what='nani')
    pass