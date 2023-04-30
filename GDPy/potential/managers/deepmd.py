#!/usr/bin/env python3
# -*- coding: utf-8 -*

import copy
from pathlib import Path
from typing import Union

import json

import numpy as np

from ase.calculators.calculator import Calculator

from GDPy.core.register import registers
from GDPy.potential.manager import AbstractPotentialManager


@registers.manager.register
class DeepmdManager(AbstractPotentialManager):

    name = "deepmd"

    implemented_backends = ["ase", "lammps"]

    valid_combinations = [
        ["ase", "ase"], # calculator, dynamics
        ["lammps", "ase"],
        ["lammps", "lammps"]
    ]

    #: Used for estimating uncertainty.
    _estimator = None

    def __init__(self, *args, **kwargs):
        """"""

        return
    
    def _create_calculator(self, calc_params: dict) -> Calculator:
        """Create an ase calculator.

        Todo:
            In fact, uncertainty estimation has various backends as well.
        
        """
        calc_params = copy.deepcopy(calc_params)

        # - some shared params
        command = calc_params.pop("command", None)
        directory = calc_params.pop("directory", Path.cwd())

        type_list = calc_params.pop("type_list", [])
        type_map = {}
        for i, a in enumerate(type_list):
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

        # - create specific calculator
        if self.calc_backend == "ase":
            # return ase calculator
            from deepmd.calculator import DP
            if models and type_map:
                calc = DP(model=models[0], type_dict=type_map)
            else:
                calc = None
        elif self.calc_backend == "lammps":
            from GDPy.computation.lammps import Lammps
            if models:
                pair_style = "deepmd {}".format(" ".join(models))
                pair_coeff = calc_params.pop("pair_coeff", "* *")

                pair_style_name = pair_style.split()[0]
                assert pair_style_name == "deepmd", "Incorrect pair_style for lammps deepmd..."

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
        """ generate calculator with various backends
        """
        super().register_calculator(calc_params)
        
        self.calc = self._create_calculator(self.calc_params)

        return
    
    def register_uncertainty_estimator(self, est_params_: dict):
        """Create an extra uncertainty estimator.

        This can be used when the current calculator is not capable of 
        estimating uncertainty.
        
        """
        from GDPy.computation.uncertainty import create_estimator
        self._estimator = create_estimator(est_params_, self.calc_params, self._create_calculator)

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

        # - convert dataset
        # --- custom conversion
        from GDPy.computation.utils import get_composition_from_atoms
        groups = {}
        for atoms in dataset:
            composition = get_composition_from_atoms(atoms)
            key = "".join([k+str(v) for k,v in composition])
            if key in groups:
                groups[key].append(atoms)
            else:
                groups[key] = [atoms]
        
        #for k, frames in groups.items():
        #    # - sort atoms
        #    pass
        # --- dpdata conversion
        import dpdata
        from ase.io import read, write
        train_set_dir = train_dir/"train"
        if not train_set_dir.exists():
            train_set_dir.mkdir()
        valid_set_dir = train_dir/"valid"
        if not valid_set_dir.exists():
            valid_set_dir.mkdir()
        
        cum_batchsizes = 0 # number of batchsizes for training
        for name, frames in groups.items():
            # --- NOTE: need convert forces to force
            frames_ = copy.deepcopy(frames) 
            for atoms in frames_:
                try:
                    forces = atoms.get_forces().copy()
                    del atoms.arrays["forces"]
                    atoms.arrays["force"] = forces
                except:
                    pass
                finally:
                    atoms.calc = None
            # --- get train and valid numbers
            nframes = len(frames_)
            n_train = int(nframes*self.train_split_ratio)
            n_train_batchsize = int(np.floor(n_train/self.train_batchsize["train"]))
            n_valid = int(nframes - n_train_batchsize*self.train_batchsize["train"])
            if n_valid <= 0:
                n_train_batchsize -= 1
                n_train = n_train_batchsize * self.train_batchsize["train"]
                n_valid = int(nframes - n_train)
            cum_batchsizes += n_train_batchsize
            
            # --- split train and valid
            train_indices = np.random.choice(nframes, n_train, replace=False).tolist()
            valid_indices = [x for x in range(nframes) if x not in train_indices]
            train_frames = [frames_[x] for x in train_indices]
            valid_frames = [frames_[x] for x in valid_indices]
            assert len(train_frames)+len(valid_frames)==nframes, "train_valid_split failed..."
            with open(train_set_dir/f"{name}-info.txt", "w") as fopen:
                content = "# train-valid-split\n"
                content += "{}\n".format(" ".join([str(x) for x in train_indices]))
                content += "{}\n".format(" ".join([str(x) for x in valid_indices]))
                fopen.write(content)

            # --- convert data
            write(train_set_dir/f"{name}-train.xyz", train_frames)
            dsys = dpdata.MultiSystems.from_file(
                train_set_dir/f"{name}-train.xyz", fmt="quip/gap/xyz", 
                type_map = self.train_config["model"]["type_map"]
            )
            dsys.to_deepmd_npy(train_set_dir) # prec, set_size

            write(valid_set_dir/f"{name}-valid.xyz", valid_frames)
            dsys = dpdata.MultiSystems.from_file(
                valid_set_dir/f"{name}-valid.xyz", fmt="quip/gap/xyz", 
                type_map = self.train_config["model"]["type_map"]
            )
            dsys.to_deepmd_npy(valid_set_dir) # prec, set_size

        # - check train config
        # NOTE: parameters
        #       numb_steps, seed
        #       descriptor-seed, fitting_net-seed
        #       training - training_data, validation_data
        train_config = copy.deepcopy(self.train_config)

        train_config["model"]["descriptor"]["seed"] = np.random.randint(0,10000)
        train_config["model"]["fitting_net"]["seed"] = np.random.randint(0,10000)

        data_dirs = list(str(x.resolve()) for x in (train_dir/"train").iterdir() if x.is_dir())
        train_config["training"]["training_data"]["systems"] = data_dirs
        #train_config["training"]["batch_size"] = 32

        data_dirs = list(str(x.resolve()) for x in (train_dir/"valid").iterdir() if x.is_dir())
        train_config["training"]["validation_data"]["systems"] = data_dirs

        train_config["training"]["seed"] = np.random.randint(0,10000)

        # --- calc numb_steps
        save_freq = train_config["training"]["save_freq"]
        n_checkpoints = int(np.ceil(cum_batchsizes*self.train_epochs/save_freq))
        numb_steps = n_checkpoints*save_freq

        train_config["training"]["numb_steps"] = numb_steps

        # - write
        with open(train_dir/"config.json", "w") as fopen:
            json.dump(train_config, fopen, indent=2)

        return
    
    def get_batchsize(self, train_config: dict) -> dict:
        """"""
        train_batchsize = train_config["training"]["training_data"]["batch_size"]
        valid_batchsize = train_config["training"]["validation_data"]["batch_size"]

        batchsize = dict(
            train = train_batchsize,
            valid = valid_batchsize
        )

        return batchsize

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
            models.append(str(p/"graph.pb"))
        models.sort()
        
        # --- update current calculator
        # NOTE: We dont need update calc_backend here...
        calc_params = copy.deepcopy(self.calc_params)
        #if self.calc_backend == "ase":
        #    for i, m in enumerate(models):
        #        calc_params = copy.deepcopy(self.calc_params)
        #        calc_params.update(backend=self.calc_backend)
        #        calc_params["model"] = m
        #        saved_calc_params = copy.deepcopy(calc_params)
        #        self.register_calculator(calc_params)
        #        self.calc.directory = Path.cwd()/f"c{i}"
        #    # NOTE: do not share calculator...
        #    self.register_calculator(saved_calc_params)
        #elif self.calc_backend == "lammps":
        #    calc_params.update(backend=self.calc_backend)
        #    # - set out_freq and out_file in lammps
        #    saved_calc_params = copy.deepcopy(calc_params)
        calc_params["model"] = models
        #self.register_calculator(calc_params)
        self.calc = self._create_calculator(calc_params)

        # --- update current estimator
        # TODO: deepmd has interal committee in lammps
        est_params = dict(
            committee = dict(
                models = models
            )
        )
        self.register_uncertainty_estimator(est_params)

        return

    def check_finished(self, model_path):
        """check if the training is finished"""
        converged = False
        model_path = Path(model_path)
        dpout_path = model_path / "dp.out"
        if dpout_path.exists():
            content = dpout_path.read_text()
            line = content.split('\n')[-3]
            print(line)
            #if 'finished' in line:
            #    converged = True

        return converged


if __name__ == "__main__":
    ...