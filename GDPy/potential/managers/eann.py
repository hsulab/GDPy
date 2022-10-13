#!/usr/bin/env python3
# -*- coding: utf-8 -*

import copy
from pathlib import Path

import yaml

from GDPy.potential.manager import AbstractPotentialManager


class EannManager(AbstractPotentialManager):

    name = "eann"
    implemented_backends = ["ase", "lammps"]

    valid_combinations = [
        ["ase", "ase"], # calculator, dynamics
        ["lammps", "ase"],
        ["lammps", "lammps"],
    ]

    TRAIN_INPUT_NAME = "input_nn.json"
    
    def __init__(self):

        return
    
    def register_calculator(self, calc_params, *args, **kwargs):
        """"""
        super().register_calculator(calc_params)

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
            else:
                calc = None
        elif self.calc_backend == "lammps":
            from GDPy.computation.lammps import Lammps
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
        
        self.calc = calc

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
        from GDPy.computation.utils import get_composition_from_atoms
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
        # - find subdirs
        train_dir = Path(train_dir)
        mdirs = []
        for p in train_dir.iterdir():
            if p.is_dir() and p.name.startswith("m"):
                mdirs.append(p.resolve())
        assert len(mdirs) == self.train_size, "Number of models does not equal model size..."

        # - find models
        calc_params = copy.deepcopy(self.calc_params)
        calc_params.update(backend=self.calc_backend)
        if self.calc_backend == "ase":
            models = []
            for p in mdirs:
                models.append(str(p/"eann_latest_py_DOUBLE.pt"))
            calc_params["file"] = models
        elif self.calc_backend == "lammps":
            models = []
            for p in mdirs:
                models.append(str(p/"eann_latest_lmp_DOUBLE.pt"))
            calc_params["pair_style"] = "eann {}".format(" ".join(models))
        #print("models: ", models)
        self.register_calculator(calc_params)

        return


if __name__ == "__main__":
    pass