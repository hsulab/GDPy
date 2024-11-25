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

        # TODO: No matter what user input, 
        #       the type_list is sorted alphabetically.
        type_list = calc_params.pop("type_list", [])
        assert type_list == sorted(type_list)
        type_list = sorted(type_list)

        type_charges = calc_params.pop("type_charges", [])

        # TODO: For simple classic potentials,
        #       we can define them by a dictionary.
        #       Thus, we need pair_style (type),
        #       pair_coeff (pair) without type id,
        #       and kspace_style (coul) if necessary.
        model_params = calc_params.pop("model", {})
        for k in ["type", "pair"]:
            assert k in model_params

        if self.calc_backend == "lammps":
            pair_style = model_params.get("type")
            pair_coeff =[]
            for k, v in model_params["pair"].items():
                coeff = " ".join([str(type_list.index(s)+1) for s in k.split("-")]) + "  " + v
                pair_coeff.append(coeff)
            pair_modify = model_params.get("modify", None)
            calc = Lammps(
                command=command, directory=directory,
                pair_style=pair_style,
                pair_coeff=pair_coeff,
                pair_modify=pair_modify,
                kspace_style=model_params.get("coul"),
                **calc_params
            )
            units = model_params.get("units", "metal")
            calc.units = units
            if calc.kspace_style is not None:
                calc.atom_style = "charge"
                calc.type_charges = type_charges
            calc.is_classic = True
        else:
            ...

        self.calc = calc

        return


if __name__ == "__main__":
    ...
  
