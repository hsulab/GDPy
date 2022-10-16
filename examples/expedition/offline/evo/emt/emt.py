#!/usr/bin/env python3
# -*- coding: utf-8 -*

from ase.calculators.emt import EMT

from GDPy.potential.manager import AbstractPotentialManager

class EmtManager(AbstractPotentialManager):

    name = "emt"
    implemented_backends = ["ase"]

    valid_combinations = [
        ["ase", "ase"]
    ]

    def register_calculator(self, calc_params, *args, **kwargs):
        super().register_calculator(calc_params)

        if self.calc_backend == "ase":
            calc = EMT()

        self.calc = calc

        return


if __name__ == "__main__":
    pass
