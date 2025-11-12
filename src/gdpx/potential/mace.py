#!/usr/bin/env python3
# -*- coding: utf-8 -*


import copy

from gdpx.backend.ase import CommitteeCalculator, DummyCalculator
from gdpx.utils.logio import remove_extra_stream_handlers

from .manager import BasePotentialManager
from .utils import build_a_committee_calculator, canonicalise_input_models


class MaceManager(BasePotentialManager):

    name = "mace"
    implemented_backends = ("ase", "jax", "lammps")

    valid_combinations = (
        ("ase", "ase"),
        ("lammps", "lammps"),
        ("jax", "ase"),
    )

    def register_calculator(self, calc_params, *agrs, **kwargs):
        """Register the calculator."""
        super().register_calculator(calc_params, *agrs, **kwargs)

        calc_params = copy.deepcopy(calc_params)

        type_list = calc_params.pop("type_list", [])

        type_map = {}
        for i, a in enumerate(type_list):
            type_map[a] = i

        # Check if all models exist and update the self.calc_params
        # as the potential may be used in other directories if submitted by a scheduler.
        models = canonicalise_input_models(calc_params.pop("model", []))
        self.calc_params.update(model=models)

        precision = calc_params.pop("precision", "float32")

        estimate_uncertainty = calc_params.get("estimate_uncertainty", False)

        calc = DummyCalculator()
        if self.calc_backend == "ase":
            try:
                import torch
                from mace.calculators import MACECalculator

                remove_extra_stream_handlers()
            except:
                raise ModuleNotFoundError("Please install mace and torch to use the ase interface.")
            device = "cuda" if torch.cuda.is_available() else "cpu"

            shared_params = dict(device=device, default_dtype=precision)
            params_list = []
            for m in models:
                specific_params = copy.deepcopy(shared_params)
                specific_params["model_paths"] = m
                params_list.append(specific_params)
            num_models = len(models)
            if num_models > 0:
                calc = build_a_committee_calculator(
                    MACECalculator,
                    params_list=params_list,
                    estimate_uncertainty=estimate_uncertainty,
                )
        elif self.calc_backend == "jax":
            try:
                import jax
                from mace_jax.calculators.mace import MACEJAXCalculator
            except:
                raise ModuleNotFoundError("Please install mace-jax and jax to use the jax interface.")
            raise NotImplementedError("The JAX backend for MACE is under development.")
        elif self.calc_backend == "lammps":
            from gdpx.computation.lammps import Lammps

            command = calc_params.pop("command", "lmp")

            # LAMMPS builds a periodic graph rather than treating ghost atoms
            # as independent nodes.
            pair_style = "mace no_domain_decomposition"

            num_models = len(models)
            if num_models != 1:
                raise Exception("MACE-LAMMPS only supports one model.")
            pair_coeff = f"* * {str(models[0])} " + "{type_list}"

            calc = Lammps(
                command=command,
                pair_style=pair_style,
                pair_coeff=pair_coeff,
                **calc_params,
            )
            calc.set(
                units="metal",
                atom_style="atomic",
                newton="on",
                atom_modify="map yes",
            )
        else:
            ...

        self.calc = calc

        return

    def switch_uncertainty_estimation(self, status: bool = True):
        """Switch on/off the uncertainty estimation."""
        # NOTE: Sometimes the manager loads several models and supports uncertainty
        #       by committee but the user disables it. We need change the calc to
        #       the correct one as the loaded one is just a single calculator.
        if not hasattr(self, "calc"):
            raise RuntimeError("Fail to switch uncertainty status as it does not have a calc.")

        # NOTE: make sure manager.as_dict() can have correct param
        self.calc_params["estimate_uncertainty"] = status

        # - convert calculator
        if self.calc_backend == "ase":
            if status:
                if isinstance(self.calc, CommitteeCalculator):
                    ...  # nothing to do
                else:  # reload models
                    self.register_calculator(self.calc_params)
            else:
                if isinstance(self.calc, CommitteeCalculator):
                    # TODO: save previous calc?
                    self.calc = self.calc.calcs[0]
                else:
                    ...
        elif self.calc_backend == "lammps":
            ...
        else:
            # TODO:
            # Other backends cannot have uncertainty estimation,
            # give a warning?
            ...

        return

    def remove_loaded_models(self):
        """Loaded models should be removed before any copy.deepcopy operations."""
        self.calc.reset()
        if self.calc_backend == "ase":
            if isinstance(self.calc, CommitteeCalculator):
                for c in self.calc.calcs:
                    c.models = None
            else:
                self.calc.models = None
        else:
            ...

        return


if __name__ == "__main__":
    ...
