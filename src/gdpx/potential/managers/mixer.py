#!/usr/bin/env python3
# -*- coding: utf-8 -*


import copy
from typing import List

from ase.calculators.calculator import Calculator, BaseCalculator

from ..calculators.mixer import EnhancedCalculator
from ..utils import convert_input_to_potter
from . import AbstractPotentialManager, DummyCalculator


class MixerManager(AbstractPotentialManager):

    name = "mixer"
    implemented_backends = ("ase",)

    valid_combinations = (
        # calculator, dynamics
        ("ase", "ase"),
        ("ase", "lammps"),
    )

    def __init__(self) -> None:
        """"""

        return

    def register_calculator(self, calc_params, *args, **kwargs) -> None:
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
            potter = convert_input_to_potter(potter_)
            potters.append(potter)

        # Used by ASE to determine timestep, dump_period ...
        self.potters = potters

        # try broadcasting calculators
        broadcast_index = -1
        for i, p in enumerate(potters):
            if isinstance(p.calc, BaseCalculator):
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

    def switch_uncertainty_estimation(self, status: bool = True):
        """"""
        if self.calc_backend == "ase":
            has_switched = False
            for manager in self.potters:
                if hasattr(manager, "switch_uncertainty_estimation"):
                    manager.switch_uncertainty_estimation(status=status)
                    has_switched = True
                else:
                    ...
            if has_switched:
                self.calc_params["potters"] = self.potters
                self.register_calculator(self.calc_params)
        else:
            ...

        return

    def as_dict(self) -> dict:
        """"""
        params = super().as_dict()
        params["params"]["potters"] = [p.as_dict() for p in self.potters]

        return params

    def remove_loaded_models(self, *args, **kwargs):
        """Loaded TF models should be removed before any copy.deepcopy operations."""
        if self.calc_backend == "ase":
            for potter in self.potters:
                if hasattr(potter, "remove_loaded_models"):
                    potter.remove_loaded_models()
        else:
            ...

        return

    @staticmethod
    def broadcast(manager: "MixerManager") -> List["MixerManager"]:
        """"""
        calc_params = copy.deepcopy(manager.calc_params)
        calc_params["backend"] = manager.calc_backend
        # print(f"{calc_params =}")

        # HACK: Here we use potter.calc to dertermine whether
        #       the potter is broadcastable
        broadcast_index = -1
        for i, p in enumerate(manager.potters):
            if isinstance(p.calc, BaseCalculator):
                ...
            else:  # assume it is a List of calculators
                if broadcast_index != -1:
                    raise RuntimeError(
                        f"Broadcast cannot on {broadcast_index} and {i}."
                    )
                broadcast_index = i

        if broadcast_index != -1:
            b_potter = manager.potters[broadcast_index]
            b_potter_cls = b_potter.__class__
            # print(f"{b_potter_cls =}")
            b_potters = b_potter_cls.broadcast(manager=b_potter)
            # print(f"{b_potters =}")
            num_potters = len(b_potters)
            broadcasted_params = []
            for i in range(num_potters):
                x = copy.deepcopy(calc_params)
                x["potters"][broadcast_index] = b_potters[i]
                broadcasted_params.append(x)
            managers = []
            for inp_dict in broadcasted_params:
                manager = MixerManager()
                manager.register_calculator(inp_dict)
                managers.append(manager)
        else:
            managers = [manager]

        return managers


if __name__ == "__main__":
    ...
