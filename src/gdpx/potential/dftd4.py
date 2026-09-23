#!/usr/bin/env python3
# -*- coding: utf-8 -*


from gdpx.backend.ase import DummyCalculator

from .manager import BasePotentialManager

"""Check https://dftd4.readthedocs.io

To install, use conda install dftd4 -c conda-forge.

Calculator parameters should have `method` (xc e.g. PBE).

"""


class Dftd4Manager(BasePotentialManager):

    name = "dftd4"

    implemented_backends = ("ase",)
    valid_combinations = ("ase", "ase")

    """See ASE documentation for calculator parameters.
    """

    def register_calculator(self, calc_params, *agrs, **kwargs):
        """"""
        super().register_calculator(calc_params, *agrs, **kwargs)

        calc = DummyCalculator()
        if self.calc_backend == "ase":
            from dftd4.ase import DFTD4 as calc_cls
        else:
            raise NotImplementedError(f"Unsupported backend {self.calc_backend}.")

        calc = calc_cls(**calc_params)

        self.calc = calc

        return


if __name__ == "__main__":
    ...
