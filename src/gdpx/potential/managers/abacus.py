#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pathlib

from ase.calculators.abacus import Abacus, AbacusProfile
from ase.io.abacus import read_input

from . import AbstractPotentialManager


class AbacusWrapper(Abacus):

    def reset(self):
        """Clear all information from old calculation."""

        self.atoms = None
        self.results = {}

        return


class AbacusManager(AbstractPotentialManager):

    name = "abacus"

    implemented_backends = ["abacus"]
    valid_combinations = (
        ("abacus", "abacus"),
        ("abacus", "ase"),
    )

    def register_calculator(self, calc_params: dict):
        """Register the calculator.

        The input parameters may contain values as:

            command: 'mpirun -n 2 abacus'
            template: INPUT_ABACUS
            pseudo_dir: ...
            basis_dir: ...

        """
        super().register_calculator(calc_params)

        command = calc_params.pop("command", None)

        pseudo_dir = str(pathlib.Path(calc_params.pop("pseudo_dir", None)).resolve())
        self.calc_params.update(pseudo_dir=pseudo_dir)
        basis_dir = str(pathlib.Path(calc_params.pop("basis_dir", None)).resolve())
        self.calc_params.update(basis_dir=basis_dir)

        template_fpath = str(pathlib.Path(calc_params.pop("template")).resolve())
        self.calc_params.update(basis_dir=basis_dir)
        kpts = calc_params.pop("kpts", (1, 1, 1))

        pp, basis = {}, {}
        for s, data in calc_params.get("type_info", {}).items():
            pp[s] = data["pseudo"]
            basis[s] = data["basis"]

        if self.calc_backend == "abacus":
            profile = AbacusProfile(
                command=command, pseudo_dir=pseudo_dir, basis_dir=basis_dir
            )
            calc = AbacusWrapper(profile, pp=pp, basis=basis, kpts=kpts)
            inp_params = read_input(template_fpath)
            calc.parameters.update(**inp_params)
            calc_type = calc.parameters.get("calculation", "scf")
            if calc_type != "scf":
                raise RuntimeError("ABACUS only supports `scf` for now.")

            self.calc = calc
        else:
            raise NotImplementedError(
                f"Unimplemented backend {self.calc_backend} for abacus."
            )

        return


if __name__ == "__main__":
    ...
