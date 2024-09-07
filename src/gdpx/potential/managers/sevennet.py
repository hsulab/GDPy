#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import pathlib

from . import AbstractPotentialManager, DummyCalculator


class SevenManager(AbstractPotentialManager):

    name = "seven"
    implemented_backends = [
        "ase"
    ]

    valid_combinations = (("ase", "ase"),)

    def register_calculator(self, calc_params, *args, **kwargs):
        """"""
        super().register_calculator(calc_params, *args, **kwargs)

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
        self.calc_params.update(model=models)

        calc = DummyCalculator()
        if self.calc_backend == "ase":
            try:
                import torch
                from sevenn.sevennet_calculator import SevenNetCalculator
            except:
                ...
            calc = SevenNetCalculator(model=models[0], file_type="checkpoint")
            ...
        else:
            ...

        self.calc = calc

        return


if __name__ == "__main__":
    ...
  
