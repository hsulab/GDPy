#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import copy
import pathlib
from typing import Union

from . import BasePotentialManager, DummyCalculator


def canonicalise_input_models(model: Union[str, list[str]]) -> list[str]:
    """Convert input models to a list of resolved path strings.

    Args:
        model: The input model(s).

    Returns:
        The resolved path strings. An error is raised if the model does not exist.

    """
    if not isinstance(model, list):
        assert isinstance(model, str)
        model_ = [model]
    else:
        model_ = model

    models = []
    for m in model_:
        m = pathlib.Path(m).resolve()
        if not m.exists():
            raise FileNotFoundError(f"The model {str(m)} does not exist.")
        models.append(str(m))

    return models


class MatterSimManager(BasePotentialManager):

    name = "mattersim"

    implemented_backends = ["ase"]

    valid_combinations = (("ase", "ase"),)

    def register_calculator(self, calc_params: dict, *agrs, **kwargs):
        """Register the calculator."""
        super().register_calculator(calc_params=calc_params, *agrs, **kwargs)

        calc_params = copy.deepcopy(calc_params)

        # Check if all models exist and update the self.calc_params
        # as the potential may be used in other directories if submitted by a scheduler.
        models = canonicalise_input_models(calc_params.pop("model", []))
        self.calc_params.update(model=models)

        # Set the default device and update it when torch is available.
        device = calc_params.pop("device", "cpu")

        calc = DummyCalculator()
        if self.calc_backend == "ase":
            try:
                import torch
                from mattersim.forcefield import MatterSimCalculator
            except:
                raise ModuleNotFoundError(
                    "Please install mattersim and torch to use the ase interface."
                )
            calc = MatterSimCalculator.from_checkpoint(load_path=models[0], device=device)
        else:
            ...  # Backend has already been checked.

        self.calc = calc

        return


if __name__ == "__main__":
    ...
