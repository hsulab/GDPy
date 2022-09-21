#!/usr/bin/env python3
# -*- coding: utf-8 -*

from pathlib import Path

from GDPy.potential.manager import AbstractPotentialManager

class NequipManager(AbstractPotentialManager):

    name = "nequip"
    implemented_backends = ["ase", "lammps"]

    valid_combinations = [
        ["ase", "ase"], # calculator, dynamics
        ["lammps", "lammps"],
        ["lammps", "lammps"]
    ]
    
    def __init__(self):

        return

    def register_calculator(self, calc_params):
        """"""
        super().register_calculator(calc_params)

        command = calc_params.pop("command", None)
        directory = calc_params.pop("directory", Path.cwd())
        atypes = calc_params.pop("type_list", [])

        models = calc_params.get("file", None)

        type_map = {}
        for i, a in enumerate(atypes):
            type_map[a] = i

        if self.calc_backend == "ase":
            # return ase calculator
            from nequip.ase import NequIPCalculator
            calc = NequIPCalculator.from_deployed_model(
                model_path=models
            )
        elif self.calc_backend == "lammps":
            from GDPy.computation.lammps import Lammps
            pair_style = calc_params.get("pair_style", None)
            if pair_style:
                calc = Lammps(
                    command=command, directory=directory, **calc_params
                )
                # - update several params
                calc.set(newton="off")
        
        self.calc = calc

        return
    
    def register_trainer(self, train_params_: dict):
        """"""
        super().register_trainer(train_params_)

        return
    
    def train(self, dataset=None, train_dir=Path.cwd()):
        """"""
        self._make_train_file(dataset, train_dir)

        return

    def _make_train_file(self, dataset=None, train_dir=Path.cwd()):
        """"""

        return
    
    def freeze(self, train_dir=Path.cwd()):
        """ freeze model and update current attached calc?
        """

        return


if __name__ == "__main__":
    pass