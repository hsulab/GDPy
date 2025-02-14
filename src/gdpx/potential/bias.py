#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import copy
from typing import List

from . import registers
from . import AbstractPotentialManager, DummyCalculator


"""This manager registers ALL bias calculators."""


class BiasManager(AbstractPotentialManager):

    name = "bias"
    implemented_backends = ("ase",)

    valid_combinations = (
        ("ase", "ase"),
        ("ase", "lammps"),
    )

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
        # FIXME: we need a better code structures to deal with colvar
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
                num_calcs = len(calc)
                if num_calcs == 1:
                    calc = calc[0]
            else:
                calc = bias_cls(**calc_params)
        else:
            ...

        self.calc = calc

        return

    @staticmethod
    def get_bias_cls(backend, method):
        """"""

        return registers.bias[method]

    @staticmethod
    def broadcast(manager: "BiasManager") -> List["BiasManager"]:
        """"""
        calc_params = copy.deepcopy(manager.calc_params)
        calc_params["backend"] = manager.calc_backend
        # print(f"{calc_params =}")

        bias_cls = manager.get_bias_cls(calc_params["backend"], calc_params["method"])
        if hasattr(bias_cls, "broadcast_params"):
            broadcasted_params = bias_cls.broadcast_params(calc_params)
            # print(f"{broadcasted_params =}")
            managers = []
            for inp_dict in broadcasted_params:
                manager = BiasManager()
                manager.register_calculator(inp_dict)
                managers.append(manager)
        else:
            managers = [manager]

        return managers


if __name__ == "__main__":
    ...
