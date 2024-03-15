#!/usr/bin/env python3
# -*- coding: utf-8 -*

import copy
from pathlib import Path
from typing import List

import yaml

from ase.calculators.calculator import Calculator

from .. import AbstractPotentialManager, AbstractTrainer


class BeannTrainer(AbstractTrainer):
    
    name = "beann"
    command = ""
    freeze_command = ""
    prefix = "config"

    def __init__(
        self, config: dict, type_list: List[str] = None, train_epochs: int = 200, 
        directory=".", command="train", freeze_command="freeze", 
        random_seed: int = 1112, *args, **kwargs
    ) -> None:
        super().__init__(
            config=config, type_list=type_list, train_epochs=train_epochs, 
            directory=directory, command=command, freeze_command=freeze_command, 
            random_seed=random_seed, *args, **kwargs
        )

        return
    
    def _resolve_train_command(self, *args, **kwargs):
        """python -u /users/jxu/repository/EANN/eann --config ./config.yaml train"""

        return
    
    def _resolve_freeze_command(self, *args, **kwargs):
        """python -u /users/jxu/repository/EANN/eann --config ./config.yaml freeze EANN.pth -o eann_latest_"""
        return super()._resolve_freeze_command(*args, **kwargs)
    
    @property
    def frozen_name(self):
        """"""
        return f"{self.name}.pth"
    
    def write_input(self, dataset, *args, **kwargs):
        """"""

        return
    
    def read_convergence(self) -> bool:
        """"""

        return


class BeannManager(AbstractPotentialManager):

    name = "beann"
    implemented_backends = ["ase", "lammps"]

    valid_combinations = (
        # calculator, dynamics
        ("ase", "ase"), 
        ("lammps", "ase"),
        ("lammps", "lammps"),
    )

    TRAIN_INPUT_NAME = "input_nn.json"
    
    def __init__(self):

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
            from eann.interface.ase.calculator import Eann
            if models and type_map:
                calc = Eann(
                    model=models, type_map=type_map,
                    command = command, directory=directory
                )
                calc.calc_uncertainty = True
            else:
                calc = None
        elif self.calc_backend == "lammps":
            from gdpx.computation.lammps import Lammps
            if models:
                pair_style = "eann {}".format(" ".join(models))
                # NOTE: need to specify precision
                pair_coeff = calc_params.pop("pair_coeff", "double * *")

                pair_style_name = pair_style.split()[0]
                assert pair_style_name == "eann", "Incorrect pair_style for lammps deepmd..."

                calc = Lammps(
                    command=command, directory=directory, 
                    pair_style=pair_style, pair_coeff=pair_coeff,
                    **calc_params
                )

                # - update several params
                calc.units = "metal"
                calc.atom_style = "atomic"
            else:
                calc = None

        return calc
    
    def register_calculator(self, calc_params, *args, **kwargs):
        """"""
        super().register_calculator(calc_params)

        self.calc = self._create_calculator(self.calc_params)

        return
    
    def register_trainer(self, train_params_: dict):
        """"""
        super().register_trainer(train_params_)

        return

    def _make_train_files(self, dataset=None, train_dir=Path.cwd()):
        """ make files for training
        """
        # - add dataset to config
        if not dataset:
            dataset = self.train_dataset
        assert dataset, f"No dataset has been set for the potential {self.name}."

        # TODO: for now, only List[Atoms]
        from gdpx.computation.utils import get_composition_from_atoms
        groups = {}
        for atoms in dataset:
            composition = get_composition_from_atoms(atoms)
            key = "".join([k+str(v) for k,v in composition])
            if key in groups:
                groups[key].append(atoms)
            else:
                groups[key] = [atoms]
        from ase.io import read, write
        systems = []
        dataset_dir = train_dir/"dataset"
        dataset_dir.mkdir()
        for key, frames in groups.items():
            k_dir = dataset_dir/key
            k_dir.mkdir()
            write(k_dir/"frames.xyz", frames)
            systems.append(str(k_dir.resolve()))

        dataset_config = dict(
            style = "auto",
            systems = systems,
            batchsizes = [[self.train_config["training"]["batchsize"],len(systems)]]
        )

        train_config = copy.deepcopy(self.train_config)
        train_config.update(dataset=dataset_config)

        train_config["training"]["epoch"] = self.train_epochs

        with open(train_dir/"config.yaml", "w") as fopen:
            yaml.safe_dump(train_config, fopen, indent=2)

        return
    
    def train(self, dataset=None, train_dir=Path.cwd()):
        """"""
        self._make_train_files(dataset, train_dir)
        # run command

        return
    
    def freeze(self, train_dir=Path.cwd()):
        """ freeze model and return a new calculator
            that may have a committee for uncertainty
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
            if self.calc_backend == "ase":
                models.append(str(p/"eann_latest_py_DOUBLE.pt"))
            elif self.calc_backend == "lammps":
                models.append(str(p/"eann_latest_lmp_DOUBLE.pt"))
        models.sort()

        # - update current calculator
        calc_params = copy.deepcopy(self.calc_params)
        calc_params["model"] = models
 
        self.calc = self._create_calculator(calc_params)

        # --- update current estimator
        # TODO: eann has interal committee in ase and lammps
        est_params = dict(
            committee = dict(
                models = models
            )
        )
        self.register_uncertainty_estimator(est_params)

        return


if __name__ == "__main__":
    pass