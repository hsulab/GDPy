#!/usr/bin/env python3
# -*- coding: utf-8 -*

from . import registers
from . import AbstractPotentialManager, DummyCalculator
from ..calculators.mixer import EnhancedCalculator


class MixerManager(AbstractPotentialManager):

    name = "mixer"
    implemented_backends = ["ase"]

    valid_combinations = [ 
        # calculator, dynamics
        ("ase", "ase"),
        ("ase", "lammps"),
    ]

    def __init__(self) -> None:
        """"""

        return

    def register_calculator(self, calc_params) -> None:
        """"""
        super().register_calculator(calc_params)

        # -
        potters_ = calc_params.get("potters", [])
        npotters = len(potters_)
        assert npotters > 1, "Mixer needs at least two potters."

        potters = []
        for potter_ in potters_:
            if isinstance(potter_, AbstractPotentialManager):
                potter = potter_
            else: # assume it is a dict
                name = potter_.get("name", None)
                potter = registers.create(
                    "manager", name, convert_name=True
                )
                potter.register_calculator(potter_.get("params", {}))
            potters.append(potter)
        self.potters = potters # Used by ASE to determine timestep, dump_period ...
        
        pot_calcs = [p.calc for p in potters]
        save_host = calc_params.get("save_host", True)

        calc = DummyCalculator()
        if self.calc_backend == "ase":
            calc = EnhancedCalculator(pot_calcs, save_host=save_host)
        else:
            ...
        
        self.calc = calc

        return
    
    def as_dict(self) -> dict:
        """"""
        params = super().as_dict()
        params["params"]["potters"] = [p.as_dict() for p in self.potters]

        return params

if __name__ == "__main__":
    ...
