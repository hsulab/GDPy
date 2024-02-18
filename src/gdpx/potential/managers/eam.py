#!/usr/bin/env python3
# -*- coding: utf-8 -*

import os
import pathlib
from typing import NoReturn

from . import AbstractPotentialManager, DummyCalculator


class EamManager(AbstractPotentialManager):

    name = "eam"

    implemented_backends = ["lammps"]
    valid_combinations = (
        ("lammps", "lammps")
    )

    """See LAMMPS documentation for calculator parameters.
    """

    def __init__(self, *args, **kwargs):
        """"""
        super().__init__()

        return
    
    def register_calculator(self, calc_params, *agrs, **kwargs):
        """"""
        super().register_calculator(calc_params, *agrs, **kwargs)

        calc = DummyCalculator()

        # - some shared params
        command = calc_params.pop("command", None)
        directory = calc_params.pop("directory", pathlib.Path.cwd())

        type_list = calc_params.pop("type_list", [])
        type_map = {}
        for i, a in enumerate(type_list):
            type_map[a] = i
        
        # --- model files
        model_ = calc_params.get("model", [])
        if not isinstance(model_, list):
            model_ = [model_]

        models = []
        for m in model_:
            m = pathlib.Path(m).resolve()
            if not m.exists():
                raise FileNotFoundError(f"Cant find model file {str(m)}")
            models.append(str(m))
        
        if self.calc_backend == "lammps":
            from gdpx.computation.lammps import Lammps
            if models:
                pair_style = "eam"
                pair_coeff = calc_params.pop("pair_coeff", "* *")
                pair_coeff += " {} ".format(models[0])

                pair_style_name = pair_style.split()[0]
                assert pair_style_name == "eam", "Incorrect pair_style for lammps eam..."

                calc = Lammps(
                    command=command, directory=directory, 
                    pair_style=pair_style, pair_coeff=pair_coeff,
                    **calc_params
                )
                # - update several params
                calc.units = "metal"
                calc.atom_style = "atomic"
        else:
            ...
        
        self.calc = calc

        return


if __name__ == "__main__":
    ...