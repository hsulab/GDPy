#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import copy
import pathlib

from gdpx.backend.ase import DummyCalculator

from ..manager import BasePotentialManager


def get_model_fpaths(params: dict) -> list[str]:
    """Get model file paths from a dict.

    The file paths will be converted to be absolute.
    """
    model_ = params.get("model", [])
    if not isinstance(model_, list):
        model_ = [model_]

    models = []
    for m in model_:
        m = pathlib.Path(m).resolve()
        if not m.exists():
            raise FileNotFoundError(f"Cant find model file {str(m)}")
        models.append(str(m))

    return models


class DeepmdJaxManager(BasePotentialManager):

    name: str = "deepmd_jax"

    implemented_backends = ("ase", "jax")

    valid_combinations = (
        ("ase", "ase"),
        ("ase", "deepmd_jax"),
        ("jax", "jax"),
        ("jax", "deepmd_jax"),
    )

    def register_calculator(self, calc_params: dict, *args, **kwargs) -> None:
        """generate calculator with various backends"""
        super().register_calculator(calc_params)
        calc_params = copy.deepcopy(calc_params)

        type_list = calc_params.pop("type_list", [])
        type_map = {}
        for i, a in enumerate(type_list):
            type_map[a] = i

        models = get_model_fpaths(calc_params)
        self.calc_params.update(model=models)

        calc = DummyCalculator()
        if self.calc_backend == "ase" or self.calc_backend == "jax":
            try:
                from .dpjax import DPJax
            except:
                raise ModuleNotFoundError("Please install deepmd-jax to use the jax interface.")
            # TODO: only support one model...
            if models:
                calc = DPJax(
                    model=models[0],
                    type_list=type_list,
                )
                print(f"{calc =}")
            else:
                ...  # No models.
        else:
            ...

        self.calc = calc

        return


if __name__ == "__main__":
    ...
