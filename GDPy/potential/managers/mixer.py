#!/usr/bin/env python3
# -*- coding: utf-8 -*

from GDPy.core.register import registers
from GDPy.computation.mixer import EnhancedCalculator
from ..manager import AbstractPotentialManager, DummyCalculator


class MixerManager(AbstractPotentialManager):

    name = "mixer"
    implemented_backends = ["ase"]

    valid_combinations = [
        ["ase", "ase"], # calculator, dynamics
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
        
        pot_calcs = [p.calc for p in potters]

        calc = DummyCalculator()
        if self.calc_backend == "ase":
            calc = EnhancedCalculator(pot_calcs)
        else:
            ...
        
        self.calc = calc

        return

if __name__ == "__main__":
    ...