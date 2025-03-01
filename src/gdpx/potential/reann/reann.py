#!/usr/bin/env python3
# -*- coding: utf-8 -*


import pathlib

from gdpx.backend.ase import CommitteeCalculator, DummyCalculator

from ..manager import BasePotentialManager


class ReannManager(BasePotentialManager):

    name = "reann"
    implemented_backends = ("ase",)

    valid_combinations = (("ase", "ase"),)

    def register_calculator(self, calc_params, *agrs, **kwargs):
        """"""
        super().register_calculator(calc_params, *agrs, **kwargs)

        type_list = calc_params.pop("type_list", [])

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

        precision = calc_params.pop("precision", "float32")
        assert precision in ["float32", "float64"]

        # TODO: make this a dataclass??
        #       currently, default disable uncertainty estimation
        estimate_uncertainty = calc_params.get("estimate_uncertainty", False)

        # - misc
        max_nneigh = calc_params.get("max_nneigh", 25000)

        # - parse calc_params
        calc = DummyCalculator()
        if self.calc_backend == "ase":
            try:
                import torch

                from .calculators.reann import REANN

                device = torch.device("cuda" if torch.cuda.is_available() else torch.device("cpu"))
                if precision == "float32":
                    precision = torch.float32
                elif precision == "float64":
                    precision = torch.float64
                else:
                    ...
            except:
                raise ModuleNotFoundError("Please install reann and torch to use the ase interface.")

            calcs = []
            for m in models:
                calc = REANN(
                    atomtype=type_list,
                    nn=m,
                    device=device,
                    dtype=precision,
                )
                calcs.append(calc)
            if len(calcs) == 1:
                calc = calcs[0]
            elif len(calcs) > 1:
                if estimate_uncertainty:
                    calc = CommitteeCalculator(calcs=calcs)
                else:
                    calc = calcs[0]
            else:
                ...
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
        # print(f"{self.calc}")

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

    def remove_loaded_models(self, *args, **kwargs):
        """Loaded TF models should be removed before any copy.deepcopy operations."""
        self.calc.reset()
        if self.calc_backend == "ase":
            if isinstance(self.calc, CommitteeCalculator):
                for c in self.calc.calcs:
                    c.pes = None
            else:
                self.calc.pes = None
        else:
            ...

        return


if __name__ == "__main__":
    ...
