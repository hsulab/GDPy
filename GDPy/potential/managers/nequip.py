#!/usr/bin/env python3
# -*- coding: utf-8 -*

from GDPy.potential.potential import AbstractPotential

class NequipManager(AbstractPotential):

    implemented_backends = ["ase"]
    valid_combinations = [
        ["ase", "ase"], # calculator, dynamics
        ["lammps", "lammps"]
    ]
    
    def __init__(self):
        pass

    def register_calculator(self, calc_params):
        """"""
        self.calc_params = calc_params

        backend = calc_params["backend"]
        if backend not in self.implemented_backends:
            raise RuntimeError()

        command = calc_params["command"]
        directory = calc_params["directory"]
        models = calc_params["file"]
        atypes = calc_params["type_list"]

        type_map = {}
        for i, a in enumerate(atypes):
            type_map[a] = i

        if backend == "ase":
            # return ase calculator
            from nequip.ase import NequIPCalculator
            calc = NequIPCalculator.from_deployed_model(
                model_path=models
            )
        elif backend == "lammps":
            pass
        
        self.calc = calc

        return


if __name__ == "__main__":
    pass