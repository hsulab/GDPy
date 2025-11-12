#!/usr/bin/env python3
# -*- coding: utf-8 -*


import copy
import importlib
import importlib.util
from typing import Union

from ase.calculators.calculator import Calculator

from gdpx.backend.ase import CommitteeCalculator, DummyCalculator
from gdpx.computation.lammps import Lammps
from gdpx.utils.logio import remove_extra_stream_handlers

from .manager import BasePotentialManager
from .utils import build_a_committee_calculator, canonicalise_input_models

try:
    from mace.calculators import MACECalculator as MACELike
except:

    class MACECalculatorStub(Calculator):
        """Placeholder MACECalculator class when mace is not installed."""

        #: The placeholder of the model need remove in remove_loaded_models.
        models = None

    MACELike = MACECalculatorStub

CalcType = Union[DummyCalculator, CommitteeCalculator, MACELike]


class MaceManager(BasePotentialManager[CalcType]):

    name = "mace"

    implemented_backends = ("ase", "jax", "lammps")

    valid_combinations = (
        ("ase", "ase"),
        ("lammps", "lammps"),
        ("jax", "ase"),
    )

    _calc: CalcType

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
            spec_macec = importlib.util.find_spec("mace.calculators")
            spec_torch = importlib.util.find_spec("torch")
            if spec_macec is None or spec_torch is None:
                raise ModuleNotFoundError("Please install mace and torch to use the ase interface.")
            else:
                ...

            torch = importlib.import_module("torch")
            MACECalculator = importlib.import_module("mace.calculators").MACECalculator

            remove_extra_stream_handlers()

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
            spec_jax = importlib.util.find_spec("jax")
            spec_macejax = importlib.util.find_spec("mace_jax.calculators")
            if spec_macejax is None or spec_jax is None:
                raise ModuleNotFoundError("Please install mace-jax and jax to use the jax interface.")
            else:
                ...

            # MACEJAXCalculator = importlib.import_module("mace_jax.calculators.mace").MACEJAXCalculator

            raise NotImplementedError("The JAX backend for MACE is under development.")
        elif self.calc_backend == "lammps":

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
        # Sometimes the manager loads several models and supports uncertainty by committee
        # but the user disables it. We need change the calc to the correct one as the loaded
        # one is just a single calculator.
        if not hasattr(self, "calc"):
            raise RuntimeError("Fail to switch uncertainty status as it does not have a calc.")

        # Make sure manager.as_dict() can have correct param
        self.calc_params["estimate_uncertainty"] = status

        # Convert calculator
        if self.calc_backend == "ase":
            if status:
                if isinstance(self.calc, CommitteeCalculator):
                    ...  # nothing to do
                else:  # reload models
                    self.register_calculator(self.calc_params)
            else:
                if isinstance(self.calc, CommitteeCalculator):
                    self.calc = self.calc.mixer.calcs[0]
                else:
                    ...
        elif self.calc_backend == "lammps":
            raise NotImplementedError("Uncertainty estimation switching is not supported for LAMMPS backend.")
        else:
            ...

        return

    def remove_loaded_models(self):
        """Loaded models should be removed before any copy.deepcopy operations."""
        self.calc.reset()
        if self.calc_backend == "ase":
            if isinstance(self.calc, DummyCalculator):
                ...
            elif isinstance(self.calc, CommitteeCalculator):
                for c in self.calc.mixer.calcs:
                    c.models = None
            else:
                self.calc.models = None
        else:
            ...

        return


if __name__ == "__main__":
    ...
