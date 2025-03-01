#!/usr/bin/env python3
# -*- coding: utf-8 -*


import copy

from gdpx.backend.ase import DummyCalculator

from .manager import BasePotentialManager
from .utils import build_a_committee_calculator, canonicalise_input_models


class NequipManager(BasePotentialManager):

    name = "nequip"
    implemented_backends = ("ase", "lammps")

    valid_combinations = (
        ("ase", "ase"),
        ("lammps", "ase"),
        ("lammps", "lammps"),
    )

    def register_calculator(self, calc_params, *args, **kwargs):
        """Register the calculator."""
        super().register_calculator(calc_params, *args, **kwargs)

        calc_params = copy.deepcopy(calc_params)

        type_list = calc_params.pop("type_list", [])

        type_map = {}
        for i, a in enumerate(type_list):
            type_map[a] = i

        # Check if all models exist and update the self.calc_params
        # as the potential may be used in other directories if submitted by a scheduler.
        models = canonicalise_input_models(calc_params.pop("model", []))
        self.calc_params.update(model=models)

        estimate_uncertainty = calc_params.get("estimate_uncertainty", False)

        calc = DummyCalculator()
        if self.calc_backend == "ase":
            try:
                import torch
                from nequip.ase import NequIPCalculator  # type: ignore
            except:
                raise ModuleNotFoundError("Please install nequip and torch to use the ase interface.")
            device = "cuda" if torch.cuda.is_available() else "cpu"

            shared_params = dict(species_to_type_name={k: k for k in type_list}, device=device)
            params_list = []
            for m in models:
                specific_params = copy.deepcopy(shared_params)
                specific_params["model_path"] = m
                params_list.append(specific_params)
            num_models = len(models)
            if num_models > 0:
                calc = build_a_committee_calculator(
                    NequIPCalculator.from_deployed_model, params_list, estimate_uncertainty=estimate_uncertainty
                )
        elif self.calc_backend == "lammps":
            from gdpx.computation.lammps import Lammps

            command = calc_params.pop("command", None)

            flavour = calc_params.pop("flavour", "nequip")  # nequip or allegro
            if models:
                pair_style = f"{flavour}"
                pair_coeff = f"* * {str(models[0])}" + " {type_list}"
                calc = Lammps(
                    command=command,
                    pair_style=pair_style,
                    pair_coeff=pair_coeff,
                    **calc_params,
                )
                # Update several extra parameters
                calc.set(units="metal", atom_style="atomic")
                if pair_style == "nequip":
                    calc.set(newton="off")
                elif pair_style == "allegro":
                    calc.set(newton="on")
                else:
                    raise Exception(f"Unknown flavour {flavour} that must be nequip or allegro.")
        else:
            ...

        return


if __name__ == "__main__":
    ...
