#!/usr/bin/env python3
# -*- coding: utf-8 -*

from pathlib import Path

from . import AbstractPotentialManager


class ReaxManager(AbstractPotentialManager):

    name = "reax"
    implemented_backends = ["lammps"]

    valid_combinations = (
        ("lammps", "ase"),
        ("lammps", "lammps")
    )

    def __init__(self, *args, **kwargs):
        """"""

        return

    def register_calculator(self, calc_params, *agrs, **kwargs):
        """"""
        super().register_calculator(calc_params, *agrs, **kwargs)

        command = calc_params.pop("command", None)
        directory = calc_params.pop("directory", Path.cwd())

        model = calc_params.get("model", None)
        model = str(Path(model).resolve())

        if self.calc_backend == "lammps":
            from gdpx.computation.lammps import Lammps
            if model:
                pair_style = "reax/c NULL"
                pair_coeff = f"* * {model}"
                calc = Lammps(
                    command=command, directory=directory, 
                    pair_style=pair_style, pair_coeff=pair_coeff,
                    **calc_params
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