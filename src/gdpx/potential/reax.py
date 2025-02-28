#!/usr/bin/env python3
# -*- coding: utf-8 -*


import pathlib

from .manager import BasePotentialManager


class ReaxManager(BasePotentialManager):

    name = "reax"

    implemented_backends = ("lammps",)
    valid_combinations = (
        ("lammps", "ase"),
        ("lammps", "lammps"),
    )

    def register_calculator(self, calc_params, *agrs, **kwargs):
        """"""
        super().register_calculator(calc_params, *agrs, **kwargs)

        command = calc_params.pop("command", None)

        model = calc_params.get("model", None)
        model = str(pathlib.Path(model).resolve())
        self.calc_params["model"] = model

        if self.calc_backend == "lammps":
            from gdpx.computation.lammps import Lammps

            if model:
                pair_style = "reax/c NULL"
                pair_coeff = f"* * {model}"
                calc = Lammps(
                    command=command,
                    pair_style=pair_style,
                    pair_coeff=pair_coeff,
                    **calc_params,
                )
                # - update several params
                calc.set(units="real")
                calc.set(atom_style="charge")
            else:
                calc = None
        else:
            ...  # The backend has already been checked.

        self.calc = calc

        return


if __name__ == "__main__":
    ...
