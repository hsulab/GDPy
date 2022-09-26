#!/usr/bin/env python3
# -*- coding: utf-8 -*

from pathlib import Path

from GDPy.potential.manager import AbstractPotentialManager

class ReaxManager(AbstractPotentialManager):

    name = "reax"
    implemented_backends = ["lammps"]

    valid_combinations = [
        ["lammps", "ase"],
        ["lammps", "lammps"]
    ]

    def __init__(self, *args, **kwargs):
        """"""

        return

    def register_calculator(self, calc_params, *agrs, **kwargs):
        """"""
        super().register_calculator(calc_params, *agrs, **kwargs)

        command = calc_params.pop("command", None)
        directory = calc_params.pop("directory", Path.cwd())

        if self.calc_backend == "lammps":
            from GDPy.computation.lammps import Lammps
            pair_style = calc_params.get("pair_style", None)
            if pair_style:
                calc = Lammps(
                    command=command, directory=directory, **calc_params
                )
                # - update several params
                calc.set(units="real")
                calc.set(atom_style="charge")
            else:
                calc = None
        self.calc = calc

        return


if __name__ == "__main__":
    pass