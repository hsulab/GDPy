#!/usr/bin/env python3
# -*- coding: utf-8 -*


from . import AbstractPotentialManager, DummyCalculator


"""Check https://dftd3.readthedocs.io/en/latest/api/ase.html

To install, use conda install dftd3-python -c conda-forge.

Calculator parameters should have `method` (xc e.g. PBE) and `damping` (e.g. d3bj).

"""


class Dftd3Manager(AbstractPotentialManager):

    name = "dftd3"

    implemented_backends = ["ase"]
    valid_combinations = (
        ("ase", "ase")
    )

    """See ASE documentation for calculator parameters.
    """

    def __init__(self, *args, **kwargs):
        """"""
        super().__init__()

        return
    
    def register_calculator(self, calc_params, *agrs, **kwargs):
        """"""
        super().register_calculator(calc_params, *agrs, **kwargs)

        calc = DummyCalculator()
        if self.calc_backend == "ase":
            from dftd3.ase import DFTD3 as calc_cls
        else:
            raise NotImplementedError(f"Unsupported backend {self.calc_backend}.")
        
        calc = calc_cls(**calc_params)

        self.calc = calc

        return


if __name__ == "__main__":
    ...