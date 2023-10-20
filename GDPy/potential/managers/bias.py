#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from GDPy.core.register import registers
from ..manager import AbstractPotentialManager, DummyCalculator
from GDPy.bias import bias_register

"""This manager registers ALL jax-based bias calculators."""


class BiasManager(AbstractPotentialManager):

    name = "bias"
    implemented_backends = ["ase"]

    valid_combinations = [
        ["ase", "ase"],
        ["ase", "lammps"],
    ]

    def __init__(self) -> None:
        """"""
        super().__init__()

        return
    
    def register_calculator(self, calc_params, *agrs, **kwargs) -> None:
        """"""
        super().register_calculator(calc_params, *agrs, **kwargs)

        # - parse params
        bias_type = calc_params.get("type", None)
        assert bias_type is not None, "Bias must have a type."

        # - instantiate calculator
        calc = DummyCalculator()
        if self.calc_backend == "ase":
            calc = bias_register[bias_type](**calc_params)
        else:
            ...
        
        self.calc = calc

        return


if __name__ == "__main__":
    ...