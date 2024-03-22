#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import copy

from . import registers
from . import AbstractPotentialManager, DummyCalculator
from gdpx.bias import bias_register

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

        # -- check whether have a colvar key
        colvar_ = None
        for k, v in calc_params.items():
            if k == "colvar":
                cv_params = copy.deepcopy(v)
                cv_name = cv_params.pop("name")
                colvar_ = registers.create("colvar", cv_name, **cv_params)
                break
        else:
            ...
        if colvar_ is not None:
            calc_params["colvar"] = colvar_

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