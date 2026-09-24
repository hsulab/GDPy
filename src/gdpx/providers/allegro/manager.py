"""Allegro potential through direct ASE inference or its LAMMPS pair style."""

import copy

from ..manager_base import BasePotentialManager
from ..potential_utils import build_a_committee_calculator, canonicalise_input_models


class AllegroManager(BasePotentialManager):
    name = "allegro"
    implemented_backends = ("ase", "lammps")
    valid_combinations = (("ase", "ase"), ("lammps", "ase"), ("lammps", "lammps"))

    def register_calculator(self, calc_params, *args, **kwargs):
        params = copy.deepcopy(calc_params)
        if "flavour" in params:
            raise ValueError("Allegro is selected by potential.provider: allegro; remove parameters.flavour.")
        super().register_calculator(params, *args, **kwargs)

        models = canonicalise_input_models(params.pop("model", []))
        if not models:
            raise ValueError("Allegro requires an exported model file.")
        self.calc_params.update(model=models)
        type_list = params.pop("type_list", None)
        estimate_uncertainty = params.pop("estimate_uncertainty", False)
        if self.calc_backend == "ase":
            try:
                from nequip.integrations.ase import NequIPCalculator
            except ImportError as exc:
                raise ModuleNotFoundError(
                    "Allegro backend ase requires NequIP with "
                    "nequip.integrations.ase.NequIPCalculator and its compiled-model API."
                ) from exc
            params.setdefault("device", "cpu")
            if type_list is not None:
                params.setdefault("chemical_species_to_atom_type_map", {symbol: symbol for symbol in type_list})
            params_list = [dict(copy.deepcopy(params), compile_path=model) for model in models]
            self.calc = build_a_committee_calculator(
                NequIPCalculator.from_compiled_model, params_list,
                estimate_uncertainty=estimate_uncertainty,
            )
        elif self.calc_backend == "lammps":
            from gdpx.providers.lammps.execution import Lammps

            # LAMMPS evaluates the first model, as in the original Allegro route.
            self.calc = Lammps(
                command=params.pop("command", "lmp"),
                pair_style="allegro",
                pair_coeff=f"* * {models[0]} {{type_list}}",
                **params,
            )
            self.calc.set(units="metal", atom_style="atomic", newton="on")
