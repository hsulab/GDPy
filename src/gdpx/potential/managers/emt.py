#!/usr/bin/env python3
# -*- coding: utf-8 -*

import os
import pathlib
from typing import NoReturn

from gdpx.core.register import registers
from gdpx.potential.manager import AbstractPotentialManager

@registers.manager.register
class EmtManager(AbstractPotentialManager):

    name = "emt"

    implemented_backends = ["ase"]
    valid_combinations = (
        ("ase", "ase"), 
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

        if self.calc_backend == "ase":
            from ase.calculators.emt import EMT
            calc_cls = EMT
        else:
            raise NotImplementedError(f"Unsupported backend {self.calc_backend}.")
        
        calc = calc_cls(**calc_params)

        self.calc = calc

        return


if __name__ == "__main__":
    ...
