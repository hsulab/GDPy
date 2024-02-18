#!/usr/bin/env python3
# -*- coding: utf-8 -*

from pathlib import Path

from . import AbstractPotentialManager


class LaspManager(AbstractPotentialManager):

    name = "lasp"
    implemented_backends = ["lasp"]
    valid_combinations = (
        ("lasp", "lasp"), # calculator, dynamics
        ("lasp", "ase")
    )

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

        command = calc_params.pop("command", None)
        directory = calc_params.pop("directory", Path.cwd())
        atypes = calc_params.pop("type_list", [])

        # --- model files
        model_ = calc_params.get("model", [])
        if not isinstance(model_, list):
            model_ = [model_]

        models = []
        for m in model_:
            m = Path(m).resolve()
            if not m.exists():
                raise FileNotFoundError(f"Cant find model file {str(m)}")
            models.append(str(m))
        
        # update to resolved paths
        self.calc_params["model"] = models
        
        pot = {}
        if len(models) == len(atypes):
            for t, m in zip(atypes,models):
                pot[t] = m
        else:
            # use first model for all types
            for t in atypes:
                pot[t] = models[0]

        self.calc = None
        if self.calc_backend == "lasp":
            from gdpx.computation.lasp import LaspNN
            self.calc = LaspNN(
                command=command, directory=directory, pot=pot,
                **calc_params
            )
        elif self.calc_backend == "lammps":
            # TODO: add lammps calculator
            pass
        else:
            raise NotImplementedError(f"{self.name} does not have {self.calc_backend}.")

        return


if __name__ == "__main__":
    pass