#!/usr/bin/env python3
# -*- coding: utf-8 -*

import copy
from pathlib import Path

import yaml

from GDPy.potential.potential import AbstractPotential
from GDPy.scheduler.factory import create_scheduler


class EannManager(AbstractPotential):

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
    
    def __parse_models(self):
        """"""
        if isinstance(self.models, str):
            pot_path = Path(self.models)
            pot_dir, pot_pattern = pot_path.parent, pot_path.name
            models = []
            for pot in pot_dir.glob(pot_pattern):
                models.append(str(pot))
            self.models = models
        else:
            for m in self.models:
                if not Path(m).exists():
                    raise ValueError('Model %s does not exist.' %m)

        return
    
    def __check_uncertainty_support(self):
        """"""
        self.uncertainty = False
        if len(self.models) > 1:
            self.uncertainty = True

        return
    
    def register_calculator(self, calc_params, *args, **kwargs):
        """"""
        super().register_calculator(calc_params)

        command = calc_params.pop("command", None)
        directory = calc_params.pop("directory", Path.cwd())
        atypes = calc_params.pop("type_list", [])

        models = calc_params.pop("file", None)
        pair_style = calc_params.get("pair_style", None)

        type_map = {}
        for i, a in enumerate(atypes):
            type_map[a] = i

        if self.calc_backend == "ase":
            # return ase calculator
            from eann.interface.ase.calculator import Eann
            if models:
                calc = Eann(
                    model=models, type_map=type_map,
                    command = command, directory=directory
                )
            else:
                calc = None
        elif self.calc_backend == "lammps":
            from GDPy.computation.lammps import Lammps
            if pair_style:
                calc = Lammps(command=command, directory=directory, **calc_params)

                # TODO: assert unit and atom_style
                #content = "units           metal\n"
                #content += "atom_style      atomic\n"
                #content = "neighbor        0.0 bin\n"
                #content += "pair_style      eann %s \n" \
                #    %(' '.join([m for m in models]))
                #content += "pair_coeff * * double %s" %(" ".join(atypes))
                #calc = content
            else:
                calc = None
        
        self.calc = calc

        return
    
    def register_trainer(self, train_params_: dict):
        """"""
        train_params = copy.deepcopy(train_params_)
        self.train_config = train_params.get("config", None)

        self.train_size = train_params.get("size", 1)
        self.train_dataset = train_params.get("dataset", None)

        scheduelr_params = train_params.get("scheduler", {}) 
        self.train_scheduler = create_scheduler(scheduelr_params)

        train_command = train_params.get("train", None)
        self.train_command = train_command

        freeze_command = train_params.get("freeze", None)
        self.freeze_command = freeze_command

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