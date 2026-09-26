#!/usr/bin/env python3
# -*- coding: utf-8 -*


import copy
import pathlib

from ..manager_base import BasePotentialManager


class ReaxManager(BasePotentialManager):

    name = "reax"

    implemented_backends = ("xreac", "reax/c")
    valid_combinations = (
        ("xreac", "ase"),
        ("reax/c", "lammps"),
        ("reax/c", "ase"),
    )

    def register_calculator(self, calc_params, *args, **kwargs):
        """Create a ReaxFF calculator using the selected backend."""
        calc_params = copy.deepcopy(calc_params)
        super().register_calculator(calc_params, *args, **kwargs)
        model = calc_params.pop("model", None)
        if not isinstance(model, str) or not model.strip():
            raise ValueError("ReaxFF model must be a non-empty path or bundled:<filename>.")

        if self.calc_backend == "xreac":
            try:
                from xreac import ForceField
                from xreac.ase import ReaxFFCalculator
            except ImportError as exc:
                raise ModuleNotFoundError(
                    "Install GDPy's reax extra (pip install 'gdpx[reax]') to use xreac."
                ) from exc
            if model.startswith("bundled:"):
                force_field = ForceField.bundled(model.removeprefix("bundled:"))
            else:
                model = str(pathlib.Path(model).expanduser().resolve())
                force_field = ForceField.from_file(model)
            calc = ReaxFFCalculator(force_field, **calc_params)
        elif self.calc_backend == "reax/c":
            from gdpx.providers.lammps.execution import Lammps

            if model.startswith("bundled:"):
                raise ValueError("The reax/c backend requires a local force-field path.")
            model = str(pathlib.Path(model).expanduser().resolve())
            calc = Lammps(
                command=calc_params.pop("command", None),
                pair_style="reax/c NULL",
                pair_coeff=f"* * {model}",
                **calc_params,
            )
            calc.set(units="real", atom_style="charge")
        self.calc_params["model"] = model
        self.calc = calc
