#!/usr/bin/env python3
# -*- coding: utf-8 -*

import os
import pathlib
from typing import NoReturn

from . import AbstractPotentialManager


class XtbManager(AbstractPotentialManager):

    name = "xtb"

    implemented_backends = ["xtb", "tblite"]
    valid_combinations = (
        ("xtb", "ase"),
        ("tblite", "ase"),
    )

    """See XTB documentation for calculator parameters.

    method: "GFN2-xTB"
    accuracy: 1.0
    electronic_temperature: 300.0
    max_iterations: 250
    solvent: "none"
    cache_api: True

    """

    def __init__(self, *args, **kwargs):
        """"""
        super().__init__()

        return
    
    def register_calculator(self, calc_params, *agrs, **kwargs):
        """"""
        super().register_calculator(calc_params, *agrs, **kwargs)

        if self.calc_backend == "xtb":
            try:
                from xtb.ase.calculator import XTB
            except:
                print("Please install xtb python to use this module.")
                exit()
            calc_cls = XTB
        elif self.calc_backend == "tblite":
            try:
                from tblite.ase import TBLite
            except:
                print("Please install xtb python to use this module.")
                exit()
            calc_cls = TBLite
        else:
            raise NotImplementedError(f"Unsupported backend {self.calc_backend}.")
        
        calc = calc_cls(**calc_params)

        self.calc = calc

        return


if __name__ == "__main__":
    ...
