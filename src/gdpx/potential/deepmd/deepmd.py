#!/usr/bin/env python3
# -*- coding: utf-8 -*


import copy
import importlib.util
from typing import Optional, Union

from ase.calculators.calculator import Calculator
from ase.data import atomic_numbers, covalent_radii

from gdpx.backend.ase import CommitteeCalculator, DummyCalculator
from gdpx.computation.lammps import Lammps

from ..manager import BasePotentialManager
from ..utils import build_a_committee_calculator, canonicalise_input_models, canonicalise_plumed_for_lammps

try:
    from .calculator import DP as DPv2
    from .calculator_v3 import DP as DPv3

    DPLike = Union[DPv2, DPv3]
except:

    class DPStub(Calculator):
        """Placeholder DeepMD class when deepmd-kit is not installed."""

        #: The placeholder of the model need remove in remove_loaded_models.
        dp = None

    DPLike = DPStub

CalcType = Union[DummyCalculator, CommitteeCalculator, Lammps, DPLike]


class DeepmdManager(BasePotentialManager[CalcType]):

    name = "deepmd"

    implemented_backends = ("ase", "jax", "lammps")

    valid_combinations = (
        ("ase", "ase"),
        ("jax", "ase"),
        ("jax", "jax"),
        ("lammps", "ase"),
        ("lammps", "lammps"),
    )

    def register_calculator(self, calc_params: dict, *args, **kwargs) -> None:
        """generate calculator with various backends"""
        super().register_calculator(calc_params=calc_params, *args, **kwargs)

        calc_params = copy.deepcopy(calc_params)

        # Some backends need a command for an external executable.
        command = calc_params.pop("command", None)

        # Check type list as early versions of deepmd need explicitly
        # set this.
        type_list = calc_params.pop("type_list", [])
        type_map = {}
        for i, a in enumerate(type_list):
            type_map[a] = i

        # Some parameters for large models
        head = calc_params.pop("head", None)

        # Check if all models exist and update the self.calc_params
        # as the potential may be used in other directories if submitted by a scheduler.
        models = canonicalise_input_models(calc_params.pop("model", []))
        self.calc_params.update(model=models)

        estimate_uncertainty = calc_params.get("estimate_uncertainty", False)

        calc = DummyCalculator()
        if self.calc_backend == "ase":
            if importlib.util.find_spec("deepmd") is None:
                raise ModuleNotFoundError("Please install deepmd-kit to use the ase interface.")

            # We need check deepmd version to decide which calculator to use.
            # Some releases do not have _version, thus, fall back to v2, for example, v2.2.10.
            dp_version = "2"
            if importlib.util.find_spec("deepmd._version") is not None:
                dp_version = getattr(importlib.import_module("deepmd._version"), "version", "2")

            if dp_version.startswith("2"):
                from .calculator import DP
            elif dp_version.startswith("3"):
                from .calculator_v3 import DP
            else:
                raise Exception(f"Unknown deepmd version {dp_version}.")

            shared_params = dict(type_dict=type_map)
            if head is not None:
                shared_params["head"] = head
            params_list = []
            for m in models:
                specific_params = copy.deepcopy(shared_params)
                params_list.append(dict(model=m, **specific_params))

            num_models = len(models)
            if num_models > 0:
                calc = build_a_committee_calculator(
                    DP,
                    params_list=params_list,
                    estimate_uncertainty=estimate_uncertainty,
                )

        elif self.calc_backend == "lammps":
            # We only need the executable path of lammps and
            # the rest of command will be completed by itself.
            # The `lmp` will be `lmp -in in.lammps 2>&1 > lmp.out`.
            if command is None:
                command = "lmp"

            pair_repulsion = calc_params.pop("pair_repulsion", {})

            pair_dispersion = calc_params.pop("pair_dispersion", {})

            if models:
                if len(models) == 1:
                    pair_style = "deepmd {}".format(" ".join(models))
                else:
                    if estimate_uncertainty:
                        pair_style = "deepmd {}".format(" ".join(models))
                    else:
                        pair_style = "deepmd {}".format(models[0])
                pair_style += " out_freq {out_freq}"

                pair_coeff = calc_params.pop("pair_coeff", "* *")
                pair_coeff += " {type_list}"

                pair_style_name = pair_style.split()[0]
                assert pair_style_name == "deepmd", "Incorrect pair_style for lammps deepmd..."

                # TODO: This is a temporary workaround for pair repulsion,
                #       it is better we use mixer to deal with a more general case.
                if pair_repulsion:
                    if pair_dispersion:
                        raise Exception("Cannot use both pair_repulsion and pair_dispersion.")
                    # See https://docs.lammps.org/pair_morse.html
                    pair_repulsion_name = pair_repulsion.get("name", "morse")
                    if pair_repulsion_name != "morse":
                        raise Exception("Only morse is supported for now.")
                    pair_repulsion_params = pair_repulsion.get("params", {})
                    d0 = pair_repulsion_params.get("d0", 1.0)  # D0, [eV]
                    alpha = pair_repulsion_params.get("alpha", 6.0)  # [1/Ang]
                    bond_ratio = pair_repulsion_params.get("cov_min", 0.8)  # the minimum ratio for the covalent bond
                    pair_style = "hybrid/overlay " + pair_style + f" {pair_repulsion_name} 3.0"
                    pair_coeff = "* * deepmd {type_list}\n"
                    # Lammps calculator uses an alphabetically order to map element to digits
                    sorted_type_list = sorted(type_list)
                    num_atypes = len(sorted_type_list)
                    for i in range(1, num_atypes + 1):
                        for j in range(i, num_atypes + 1):
                            r0 = (
                                covalent_radii[atomic_numbers[sorted_type_list[i - 1]]]
                                + covalent_radii[atomic_numbers[sorted_type_list[j - 1]]]
                            ) * bond_ratio
                            pair_coeff += f"pair_coeff  {i} {j} morse {d0} {alpha} {r0:>.2f} {r0:>.2f}\n"

                if pair_dispersion:
                    if pair_repulsion:
                        raise Exception("Cannot use both pair_repulsion and pair_dispersion.")
                    # See https://docs.lammps.org/pair_dispersion_d3.html
                    dispersion_name = pair_dispersion.get("name", "d3")
                    if dispersion_name != "d3":
                        raise Exception("Only d3 is supported for now.")
                    method = pair_dispersion.get("method", "pbe")  # functional
                    damping = pair_dispersion.get("damping", "bj")  # damping, original, zerom, bj, bjm
                    r_cut = pair_dispersion.get("r_cut", 30.0)  # [Ang]
                    r_cn_cut = pair_dispersion.get("r_cn_cut", 20.0)  # [Ang]
                    # To construct,
                    pair_style = (
                        "hybrid/overlay "
                        + pair_style
                        + f" dispersion/{dispersion_name} {damping} {method} {r_cut} {r_cn_cut}"
                    )
                    pair_coeff = "* * deepmd {type_list}\n"
                    pair_coeff += f"pair_coeff  * * dispersion/{dispersion_name}" + " {type_list}\n"

                # Initialise the LAMMPS calculator
                calc = Lammps(
                    command=command,
                    pair_style=pair_style,
                    pair_coeff=pair_coeff,
                    **calc_params,
                )
                # Update several extra parameters,
                # we must use set method, otherwise, __getattribute__ is used
                # instead of __getattr__.
                calc.set(
                    units="metal",
                    atom_style="atomic",
                    neighbor="2.0 bin",
                    neigh_modify="every 10 check yes",
                )

                # Check if auxiliary bias is provided
                aux_dict = calc_params.pop("aux", {})
                if aux_dict:
                    aux_name = aux_dict.get("name", "plumed")
                    if aux_name == "plumed":
                        aux_params = canonicalise_plumed_for_lammps(aux_dict["params"])
                        self.calc_params.update(aux=dict(name=aux_name, params=aux_params))
                    else:
                        raise Exception("Only plumed is supported.")
                if aux_dict:
                    calc.set(plumed=aux_dict["params"]["inp"])
            else:
                ...  # No models provided, use DummyCalculator
        else:
            ...  # The backend has already been checked.

        self.calc = calc

        return

    def switch_backend(self, backend: Optional[str] = None) -> None:
        """Switch the potential's calculation backend."""
        if backend is None:
            return

        if not hasattr(self, "calc"):
            raise RuntimeError(f"{self.name} cannot switch backend as it does not have a calculator attached.")
        if backend not in self.implemented_backends:
            raise RuntimeError(f"{self.name} cannot switch backend from {self.calc_backend} to {backend}.")

        prev_backend = self.calc_backend
        if prev_backend == "ase" and backend == "lammps":
            calc_params = copy.deepcopy(self.calc_params)
            calc_params["backend"] = "lammps"
            command = calc_params.get("command", None)
            if command is None:
                raise RuntimeError(f"{self.name} cannot switch backend from ase to lammps as no command is provided.")
            else:
                self.calc_params["backend"] = "lammps"
            self.register_calculator(calc_params)
        elif prev_backend == "lammps" and backend == "ase":
            calc_params = copy.deepcopy(self.calc_params)
            calc_params["backend"] = "ase"
            self.calc_params["backend"] = "ase"
            self.register_calculator(calc_params)
        else:  # Nothing to do for other combinations
            ...

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
            if isinstance(self.calc, Lammps):
                # The string is `pair_style deepmd m0 m1 m2 m3`
                models = self.calc.pair_style.split()[1:]
                num_models = len(models)
                if status:
                    if num_models > 1:
                        ...
                    else:
                        self.register_calculator(self.calc_params)
                else:
                    if num_models > 1:
                        self.calc.set(pair_style=f"deepmd {models[0]}")
                    else:
                        ...
            else:
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
                    c.dp = None
            elif isinstance(self.calc, Lammps):
                ...  # Lammps calculator does not load models in Python side
            else:
                self.calc.dp = None
        else:
            ...

        return


if __name__ == "__main__":
    ...
