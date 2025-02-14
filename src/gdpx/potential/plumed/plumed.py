#!/usr3/bin/env python3
# -*- coding: utf-8 -*


import pathlib

from .. import AbstractPotentialManager, DummyCalculator


class PlumedManager(AbstractPotentialManager):

    name = "plumed"

    implemented_backends = ["ase"]

    valid_combinations = (
        # calculator, dynamics
        ("ase", "ase"),
    )

    def __init__(self) -> None:
        """"""

        return

    def register_calculator(self, calc_params: dict, *agrs, **kwargs) -> None:
        """"""
        super().register_calculator(calc_params, *agrs, **kwargs)

        calc = DummyCalculator()
        if self.calc_backend == "ase":
            try:
                from .calculators.plumed2 import Plumed
            except:
                raise ModuleNotFoundError(
                    "Please install py-plumed to use the ase interface."
                )

            inp = calc_params.get("inp", "./plumed.inp")
            if isinstance(inp, str) or isinstance(inp, pathlib.Path):
                inp = pathlib.Path(inp)
                if inp.exists():
                    input_lines = []
                    with open(inp, "r") as fopen:
                        lines = fopen.readlines()
                        for line in lines:
                            line = line.strip()
                            if line and not line.startswith("#"):
                                if "#" in line:
                                    line = line[: line.index("#")]
                                else:
                                    line = line
                                input_lines.append(line + "\n")
                    self.calc_params.update(inp=input_lines)
                else:
                    raise FileNotFoundError(f"{inp} does not exist.")
            elif isinstance(inp, list):
                input_lines = inp
            else:
                raise ValueError(f"Plumed input {inp} {type(inp)} is invalid.")

            kT = calc_params.get("kT", 1.0)
            use_charge = calc_params.get("use_charge", False)
            update_charge = calc_params.get("update_charge", False)
            calc = Plumed(
                input=input_lines,
                kT=kT,
                use_charge=use_charge,
                update_charge=update_charge,
            )
        else:
            ...

        self.calc = calc

        return


if __name__ == "__main__":
    ...
