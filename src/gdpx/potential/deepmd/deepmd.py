#!/usr/bin/env python3
# -*- coding: utf-8 -*


import copy

from ase.calculators.calculator import Calculator
from ase.data import atomic_numbers, covalent_radii

from gdpx.backend.ase import CommitteeCalculator, DummyCalculator
from gdpx.utils.logio import remove_extra_stream_handlers

from ..manager import BasePotentialManager
from ..utils import build_a_committee_calculator, canonicalise_input_models


class DeepmdManager(BasePotentialManager):

    name = "deepmd"

    implemented_backends = ("ase", "jax", "lammps")

    valid_combinations = (
        ("ase", "ase"),
        ("jax", "ase"),
        ("jax", "jax"),
        ("lammps", "ase"),
        ("lammps", "lammps"),
    )

    def _create_calculator(self, calc_params: dict) -> Calculator:
        """Create an ase calculator.

        Todo:
            In fact, uncertainty estimation has various backends as well.

        """
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

        # TODO: make this a dataclass??
        #       currently, default disable uncertainty estimation
        estimate_uncertainty = calc_params.get("estimate_uncertainty", False)

        # Create a specific calculator
        calc = DummyCalculator()
        if self.calc_backend == "ase":
            try:
                import deepmd
            except:
                raise ModuleNotFoundError("Please install deepmd-kit to use the ase interface.")

            try:
                from deepmd._version import version as dp_version
            except:
                # Some releases do not have _version, thus, fall back to v2,
                # for example, v2.2.10
                dp_version = "2"

            if dp_version.startswith("2"):
                from .calculator import DP
            elif dp_version.startswith("3"):
                from .calculator_v3 import DP
            else:
                raise Exception(f"Unknown deepmd version {dp_version}.")

            remove_extra_stream_handlers()

            shared_params = dict(type_dict=type_map)
            if head is not None:
                shared_params["head"] = head
            params_list = []
            for m in models:
                specific_params = copy.deepcopy(shared_params)
                specific_params["model"] = m
                params_list.append(specific_params)
            num_models = len(models)
            if num_models > 0:
                calc = build_a_committee_calculator(
                    DP,
                    params_list=params_list,
                    estimate_uncertainty=estimate_uncertainty,
                )
        elif self.calc_backend == "lammps":
            from gdpx.computation.lammps import Lammps

            # We only need the executable path of lammps and
            # the rest of command will be completed by itself.
            # The `lmp` will be `lmp -in in.lammps 2>&1 > lmp.out`.
            if command is None:
                command = "lmp"

            pair_repulsion = calc_params.pop("pair_repulsion", {})

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
                    # See https://docs.lammps.org/pair_morse.html
                    pair_repulsion_name = pair_repulsion.get("name", "morse")
                    if pair_repulsion_name != "morse":
                        raise Exception("Only morse is supported for now.")
                    pair_repulsion_params = pair_repulsion.get("params", {})
                    d0 = pair_repulsion_params.get("d0", 1.0) # D0, [eV]
                    alpha = pair_repulsion_params.get("alpha", 6.0) # [1/Ang]
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
                            pair_coeff += (
                                f"pair_coeff  {i} {j} morse {d0} {alpha} {r0:>.2f} {r0:>.2f}\n"
                            )

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
        else:
            ...  # The backend has already been checked.

        return calc

    def register_calculator(self, calc_params, *args, **kwargs) -> None:
        """generate calculator with various backends"""
        super().register_calculator(calc_params)

        self.calc = self._create_calculator(self.calc_params)

        return

    def switch_backend(self, backend: str = None) -> None:
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
                    self.calc = self._create_calculator(self.calc_params)
            else:
                if isinstance(self.calc, CommitteeCalculator):
                    # TODO: save previous calc?
                    self.calc = self.calc.calcs[0]
                else:
                    ...
        elif self.calc_backend == "lammps":
            models = self.calc.pair_style.split()[1:]  # model paths
            nmodels = len(models)
            if status:
                if nmodels > 1:
                    ...
                else:
                    self.calc = self._create_calculator(self.calc_params)
            else:
                # TODO: use self.calc_params? It should be protected?
                # pair_style deepmd m0 m1 m2 m3
                if nmodels > 1:
                    self.calc.pair_style = f"deepmd {models[0]}"
                else:
                    ...
        else:
            # TODO:
            # Other backends cannot have uncertainty estimation,
            # give a warning?
            ...
        # print(f"{self.calc}")

        return

    def remove_loaded_models(self, *args, **kwargs):
        """Loaded TF models should be removed before any copy.deepcopy operations."""
        self.calc.reset()
        if self.calc_backend == "ase":
            if isinstance(self.calc, CommitteeCalculator):
                for c in self.calc.calcs:
                    c.dp = None
            else:
                self.calc.dp = None
        else:
            ...

        return


if __name__ == "__main__":
    ...
