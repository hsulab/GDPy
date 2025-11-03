#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import copy

from gdpx.backend.ase import DummyCalculator

from ..manager import BasePotentialManager
from ..utils import build_a_committee_calculator, canonicalise_input_models


class DeepmdJaxXManager(BasePotentialManager):

    name: str = "deepmd_jax_x"

    implemented_backends = (
        "ase",
        "lammps",
    )

    valid_combinations = (
        ("ase", "ase"),
        ("lammps", "lammps"),
    )

    def register_calculator(self, calc_params: dict, *args, **kwargs) -> None:
        """Register the calculator."""
        super().register_calculator(calc_params=calc_params, *args, **kwargs)

        calc_params = copy.deepcopy(calc_params)

        # Check if all models exist and update the self.calc_params
        # as the potential may be used in other directories if submitted by a scheduler.
        models = canonicalise_input_models(calc_params.pop("model", []))
        self.calc_params.update(model=models)

        estimate_uncertainty = calc_params.get("estimate_uncertainty", False)

        calc = DummyCalculator()
        if self.calc_backend == "ase":
            try:
                from dpjx.interface.ase import DPJax
            except:
                raise ModuleNotFoundError("Please install dpjx and jax to use the ase interface.")

            shared_params = dict()
            params_list = []
            for m in models:
                specific_params = copy.deepcopy(shared_params)
                specific_params["model"] = m
                params_list.append(specific_params)

            num_models = len(models)
            if num_models > 0:
                calc = build_a_committee_calculator(
                    DPJax,
                    params_list=params_list,
                    estimate_uncertainty=estimate_uncertainty,
                )
        elif self.calc_backend == "lammps":
            raise NotImplementedError(
                "The lammps backend is not implemented for the deepmd_jax_x potential. Please use the ase backend."
            )
        else:
            ...  # Backend has already been checked.

        self.calc = calc

        return


if __name__ == "__main__":
    ...
