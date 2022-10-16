#!/usr/bin/env python3
# -*- coding: utf-8 -*

import copy
from pathlib import Path

import yaml

import numpy as np

from ase.calculators.calculator import Calculator

from GDPy.potential.manager import AbstractPotentialManager

class NequipManager(AbstractPotentialManager):

    name = "nequip"
    implemented_backends = ["ase", "lammps"]

    valid_combinations = [
        ["ase", "ase"], # calculator, dynamics
        ["lammps", "ase"],
        ["lammps", "lammps"]
    ]
    
    def __init__(self):
        """"""
        self.committee = None

        return

    def _create_calculator(self, calc_params: dict) -> Calculator:
        """Create an ase calculator.

        Todo:
            In fact, uncertainty estimation has various backends as well.
        
        """
        calc_params = copy.deepcopy(calc_params)

        command = calc_params.pop("command", None)
        directory = calc_params.pop("directory", Path.cwd())
        atypes = calc_params.pop("type_list", [])

        type_map = {}
        for i, a in enumerate(atypes):
            type_map[a] = i

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

        if self.calc_backend == "ase":
            # return ase calculator
            from nequip.ase import NequIPCalculator
            if models:
                calc = NequIPCalculator.from_deployed_model(
                    model_path=models[0]
                )
            else:
                calc = None
        elif self.calc_backend == "lammps":
            from GDPy.computation.lammps import Lammps
            flavour = calc_params.pop("flavour", "nequip") # nequip or allegro
            if models:
                pair_style = "{}".format(flavour)
                pair_coeff = "* * {}".format(models[0])
                calc = Lammps(
                    command=command, directory=directory, 
                    pair_style=pair_style, pair_coeff=pair_coeff,
                    **calc_params
                )
                # - update several params
                calc.units = "metal"
                calc.atom_style = "atomic"
                if pair_style == "nequip":
                    calc.set(**dict(newton="off"))
                elif pair_style == "allegro":
                    calc.set(**dict(newton="on"))
            else:
                calc = None

        return calc

    def register_calculator(self, calc_params):
        """"""
        super().register_calculator(calc_params)
        
        self.calc = self._create_calculator(calc_params)

        return
    
    def register_trainer(self, train_params_: dict):
        """"""
        super().register_trainer(train_params_)
        # print(self.train_config)

        return
    
    def train(self, dataset=None, train_dir=Path.cwd()):
        """"""
        self._make_train_files(dataset, train_dir)

        return

    def _make_train_files(self, dataset=None, train_dir=Path.cwd()):
        """ make files for training
        """
        # - add dataset to config
        if not dataset: # NOTE: can be a path or a List[Atoms]
            dataset = self.train_dataset
        assert dataset, f"No dataset has been set for the potential {self.name}."

        # - check dataset
        from ase.io import read, write
        write(train_dir/"dataset.xyz", dataset)

        nframes = len(dataset)
        n_train = int(nframes*self.train_split_ratio)
        n_val = nframes - n_train

        # - check train config
        # params: root, run_name, seed, dataset_seed, n_train, n_val, batch_size
        #         dataset, dataset_file_name
        train_config = copy.deepcopy(self.train_config)

        train_config["root"] = str(train_dir.resolve())
        train_config["run_name"] = "auto"

        train_config["seed"] = np.random.randint(0,10000)
        train_config["dataset_seed"] = np.random.randint(0,10000)

        train_config["dataset"] = "ase"
        train_config["dataset_file_name"] = str((train_dir/"dataset.xyz").resolve())

        train_config["n_train"] = n_train
        train_config["n_val"] = n_val

        train_config["max_epochs"] = self.train_epochs

        with open(train_dir/"config.yaml", "w") as fopen:
            yaml.safe_dump(train_config, fopen)

        return
    
    def freeze(self, train_dir=Path.cwd()):
        """ freeze model and update current attached calc?
        """
        super().freeze(train_dir)

        # - find subdirs
        train_dir = Path(train_dir)
        mdirs = []
        for p in train_dir.iterdir():
            if p.is_dir() and p.name.startswith("m"):
                mdirs.append(p.resolve())
        assert len(mdirs) == self.train_size, "Number of models does not equal model size..."

        # - find models and form committee
        models = []
        for p in mdirs:
            models.append(str(p/"deployed_model.pth"))

        # --- update current calculator
        # NOTE: We dont need update calc_backend here...
        calc_params = copy.deepcopy(self.calc_params)
        calc_params["model"] = models
        self.calc = self._create_calculator(calc_params)

        # --- update current estimator
        est_params = dict(
            committee = dict(
                models = models
            )
        )
        self.register_uncertainty_estimator(est_params)

        return


if __name__ == "__main__":
    pass