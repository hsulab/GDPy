#!/usr/bin/env python3
# -*- coding: utf-8 -*

from pathlib import Path

from GDPy.potential.potential import AbstractPotential

class LaspManager(AbstractPotential):

    name = "lasp"
    implemented_backends = ["lasp"]
    valid_combinations = [
        ["lasp", "lasp"], # calculator, dynamics
        ["lasp", "ase"]
    ]

    def __init__(self):

        return

    def register_calculator(self, calc_params):
        """ params
            command
            directory
            pot
        """
        super().register_calculator(calc_params)

        self.calc_params["pot_name"] = self.name

        # NOTE: need resolved pot path
        pot = {}
        pot_ = calc_params.get("pot")
        for k, v in pot_.items():
            pot[k] = str(Path(v).resolve())
        self.calc_params["pot"] = pot

        self.calc = None
        if self.calc_backend == "lasp":
            from GDPy.computation.lasp import LaspNN
            self.calc = LaspNN(**self.calc_params)
        elif self.calc_backend == "lammps":
            # TODO: add lammps calculator
            pass
        else:
            raise NotImplementedError(f"{self.name} does not have {self.calc_backend}.")

        return


if __name__ == "__main__":
    pass