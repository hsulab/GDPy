#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import copy

from . import registers
from . import AbstractPotentialManager, DummyCalculator


"""This manager registers ALL bias calculators."""


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

        # parse params
        calc_params = copy.deepcopy(calc_params)
        bias_method = calc_params.pop("method", None)
        assert bias_method is not None, "Bias must have a method."

        # check whether have a colvar key
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

        # instantiate calculator
        calc = DummyCalculator()
        if self.calc_backend == "ase":
            bias_cls = registers.bias[bias_method]
            if hasattr(bias_cls, "broadcast"):
                calc = bias_cls.broadcast(calc_params)
            else:
                calc = bias_cls(**calc_params)
        else:
            ...

        self.calc = calc

        return


if __name__ == "__main__":
    ...
