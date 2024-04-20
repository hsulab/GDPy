#!/usr/bin/env python3
# -*- coding: utf-8 -*


import copy

from . import registers
from . import AbstractPotentialManager, DummyCalculator
from ..calculators.mixer import EnhancedCalculator

from ase.calculators.calculator import Calculator


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

        # some basic parameters
        save_host = calc_params.get("save_host", True)

        # process potters
        potters_ = calc_params.get("potters", [])
        npotters = len(potters_)
        assert npotters > 1, "Mixer needs at least two potters."

        potters = []
        for potter_ in potters_:
            if isinstance(potter_, AbstractPotentialManager):
                potter = potter_
            else:  # assume it is a dict
                name = potter_.get("name", None)
                potter = registers.create("manager", name, convert_name=True)
                potter.register_calculator(potter_.get("params", {}))
            potters.append(potter)
        self.potters = potters  # Used by ASE to determine timestep, dump_period ...

        # try broadcasting calculators
        broadcast_index = -1
        for i, p in enumerate(potters):
            if isinstance(p.calc, Calculator):
                ...
            else:  # assume it is a List of calculators
                if broadcast_index != -1:
                    raise RuntimeError(
                        f"Broadcast cannot on {broadcast_index} and {i}."
                    )
                broadcast_index = i

        calc = DummyCalculator()
        if self.calc_backend == "ase":
            if broadcast_index == -1:
                pot_calcs = [p.calc for p in potters]
                calc = EnhancedCalculator(pot_calcs, save_host=save_host)
            else:
                # non-broadcasted calculators are shared...
                num_instances = len(potters[broadcast_index].calc)
                new_pot_calcs = []
                for i in range(num_instances):
                    x = [p.calc for p in potters]
                    x[broadcast_index] = potters[broadcast_index].calc[i]
                    new_pot_calcs.append(x)
                calc = [
                    EnhancedCalculator(x, save_host=save_host) for x in new_pot_calcs
                ]
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
