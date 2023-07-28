#!/usr/bin/env python3
# -*- coding: utf-8 -*


import copy
import pathlib
from typing import List


from GDPy.core.register import registers
from ..manager import AbstractPotentialManager, DummyCalculator
from ..trainer import AbstractTrainer
from GDPy.computation.mixer import CommitteeCalculator


@registers.trainer.register
class MaceTrainer(AbstractTrainer):

    name = "mace"
    command = ""
    freeze_command = ""

    def __init__(
        self, config: dict, type_list: List[str] = None, train_epochs: int = 200, 
        directory=".", command="train", freeze_command="freeze", 
        random_seed: int = None, *args, **kwargs
    ) -> None:
        """"""
        super().__init__(config, type_list, train_epochs, directory, command, freeze_command, random_seed, *args, **kwargs)

        self._type_list = type_list

        return


class MaceManager(AbstractPotentialManager):

    name = "mace"
    implemented_backends = ["ase"]

    valid_combinations = [
        ["ase", "ase"]
    ]

    def __init__(self):
        """"""

        return
    
    def register_calculator(self, calc_params, *agrs, **kwargs):
        """"""
        super().register_calculator(calc_params, *agrs, **kwargs)

        # - parse params
        calc_params = copy.deepcopy(calc_params)

        command = calc_params.pop("command", None)
        directory = calc_params.pop("directory", pathlib.Path.cwd())
        type_list = calc_params.pop("type_list", [])

        type_map = {}
        for i, a in enumerate(type_list):
            type_map[a] = i

        # - model files
        model_ = calc_params.get("model", [])
        if not isinstance(model_, list):
            model_ = [model_]

        models = []
        for m in model_:
            m = pathlib.Path(m).resolve()
            if not m.exists():
                raise FileNotFoundError(f"Cant find model file {str(m)}")
            models.append(str(m))

        # - create specific calculator
        calc = DummyCalculator()
        if self.calc_backend == "ase":
            # return ase calculator
            try:
                import torch
                from mace.calculators import MACECalculator
            except:
                raise ModuleNotFoundError("Please install nequip and torch to use the ase interface.")
            calcs = []
            for m in models:
                curr_calc = MACECalculator.from_deployed_model(
                    model_path=m, 
                    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
                )
                calcs.append(curr_calc)
            if len(calcs) == 1:
                calc = calcs[0]
            elif len(calcs) > 1:
                calc = CommitteeCalculator(calcs)
            else:
                ...
        elif self.calc_backend == "lammps":
            raise RuntimeError("The LAMMPS backend for MACE is under development.")
        else:
            ...
        
        self.calc = calc

        return


if __name__ == "__main__":
    ...