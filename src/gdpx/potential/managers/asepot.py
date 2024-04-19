#!/usr/bin/env python3
# -*- coding: utf-8 -*


import copy

from . import AbstractPotentialManager


class AsePotManager(AbstractPotentialManager):

    name = "ase"

    implemented_backends = ["ase"]
    valid_combinations = (
        ("ase", "ase"),
    )

    """Here is an interface to ase built-in calculators."""

    def __init__(self, *args, **kwargs):
        """"""
        super().__init__()

        return

    def register_calculator(self, calc_params: dict, *args, **kwargs):
        """"""
        super().register_calculator(calc_params, *args, **kwargs)

        calc_params = copy.deepcopy(calc_params)
        method = calc_params.pop("method", "")

        if self.calc_backend == "ase":
            if method == "lj":
                from ase.calculators.lj import LennardJones
                calc_cls = LennardJones
            elif method == "morse":
                from ase.calculators.morse import MorsePotential
                calc_cls = MorsePotential
            elif method == "tip3p":
                from ase.calculators.tip3p import TIP3P
                calc_cls = TIP3P
            else:
                raise NotImplementedError(f"Unsupported potential {method}.")
        else:
            raise NotImplementedError(f"Unsupported backend {self.calc_backend}.")
        
        calc = calc_cls(**calc_params)

        self.calc = calc

        return


if __name__ == "__main__":
    ...
