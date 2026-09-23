#!/usr/bin/env python3
# -*- coding: utf-8 -*


import copy
from typing import Union

from ase.calculators.calculator import Calculator

from gdpx.backend.ase import CommitteeCalculator, DummyCalculator

from ..manager import BasePotentialManager
from ..utils import build_a_committee_calculator, canonicalise_input_models

try:
    from .calculators.reann import REANN as REANNLike
except:

    class REANNStub(Calculator):
        """Placeholder REANN class when reann is not installed."""

        #: The placeholder of the model need remove in remove_loaded_models.
        pes = None

    REANNLike = REANNStub


CalcType = Union[DummyCalculator, CommitteeCalculator, REANNLike]


class ReannManager(BasePotentialManager[CalcType]):
    name = "reann"

    implemented_backends = ("ase",)

    valid_combinations = (("ase", "ase"),)

    _calc: CalcType

    def register_calculator(self, calc_params: dict, *agrs, **kwargs):
        """"""
        super().register_calculator(calc_params=calc_params, *agrs, **kwargs)

        calc_params = copy.deepcopy(calc_params)

        # Check if all models exist and update the self.calc_params
        # as the potential may be used in other directories if submitted by a scheduler.
        models = canonicalise_input_models(calc_params.pop("model", []))
        self.calc_params.update(model=models)

        type_list = calc_params.pop("type_list", [])

        precision = calc_params.pop("precision", "float32")
        assert precision in ("float32", "float64")

        compute_stress = calc_params.pop("compute_stress", False)

        estimate_uncertainty = calc_params.get("estimate_uncertainty", False)

        calc = DummyCalculator()
        if self.calc_backend == "ase":
            try:
                import torch

                from .calculators.reann import REANN
            except:
                raise ModuleNotFoundError("Please install reann and torch to use the ase interface.")

            device = "cuda" if torch.cuda.is_available() else "cpu"

            # The official REANN calculator requires a fortran module to be compiled,
            # and needs max_nneigh to be specified. Here we bypass these requirements
            # by directly loading the torchscript model, and use the ase neighborlist
            # instead.
            shared_params = dict(atomtype=type_list, compute_stress=compute_stress, device=device, dtype=precision)
            params_list = []
            for m in models:
                specific_params = copy.deepcopy(shared_params)
                specific_params["nn"] = m
                params_list.append(specific_params)

            num_models = len(models)
            if num_models > 0:
                calc = build_a_committee_calculator(
                    REANN,
                    params_list=params_list,
                    estimate_uncertainty=estimate_uncertainty,
                )
        elif self.calc_backend == "lammps":
            raise NotImplementedError(
                "The lammps backend is not implemented for the reann potential. Please use the ase backend."
            )
        else:
            ...  # Backend has already been checked.

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
            ...
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
                    c.pes = None
            else:
                self.calc.pes = None
        else:
            ...

        return


if __name__ == "__main__":
    ...
