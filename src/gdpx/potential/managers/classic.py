#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import pathlib

from . import AbstractPotentialManager, DummyCalculator
from .. import Lammps


class ClassicManager(AbstractPotentialManager):

    name = "classic"

    implemented_backends = ["lammps"]
    valid_combinations = (
        ("lammps", "lammps"),
    )


    def register_calculator(self, calc_params, *args, **kwargs):
        """"""
        super().register_calculator(calc_params, *args, **kwargs)

        calc = DummyCalculator()

        # some shared params
        command = calc_params.pop("command", None)
        directory = calc_params.pop("directory", pathlib.Path.cwd())
        type_list = calc_params.pop("type_list", [])

        if self.calc_backend == "lammps":
            pair_style = "buck/coul/long 10.0"
            pair_coeff =[
                "1 1 9547.96 0.21916 32.0",
                "1 2  529.70 0.3581   0.0",
                "2 2    0.0  1.0      0.0"
            ]
            calc = Lammps(
                command=command, directory=directory,
                pair_style=pair_style,
                pair_coeff=pair_coeff,
                kspace_style="ewald  1e-4",
                **calc_params
            )
            calc.units = "metal"
            calc.atom_style = "charge"
            calc.is_classic = True
        else:
            ...

        self.calc = calc

        return


if __name__ == "__main__":
    ...
  
