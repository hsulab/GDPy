#!/usr3/bin/env python3
# -*- coding: utf-8 -*

from ..manager import AbstractPotentialManager, DummyCalculator


class PlumedManager(AbstractPotentialManager):

    name = "plumed"

    implemented_backends = ["ase"]

    valid_combinations = [
        ["ase", "ase"], # calculator, dynamics
    ]

    def __init__(self) -> None:
        """"""

        return
    
    def register_calculator(self, calc_params: dict, *agrs, **kwargs) -> None:
        """"""
        super().register_calculator(calc_params, *agrs, **kwargs)

        calc = DummyCalculator()
        if self.calc_backend == "ase":
            try:
                from GDPy.computation.plumed import Plumed
            except:
                raise ModuleNotFoundError("Please install py-plumed to use the ase interface.")
            input_lines = [
                "FLUSH STRIDE=1\n",
                "d1: DISTANCE ATOMS=1,2\n",
                "PRINT FILE=COLVAR ARG=d1 STRIDE=1\n"
            ]
            calc = Plumed(input=input_lines, log="plumed.out")
        else:
            ...
        
        self.calc = calc

        return


if __name__ == "__main__":
    ...