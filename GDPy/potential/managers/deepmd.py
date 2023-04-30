#!/usr/bin/env python3
# -*- coding: utf-8 -*

import copy
from pathlib import Path
import pathlib
from typing import Union, List

import json

import numpy as np

from ase import Atoms
from ase.io import read, write
from ase.calculators.calculator import Calculator

from GDPy.core.register import registers
from GDPy.potential.manager import AbstractPotentialManager

def group_dataset(dataset):
    """"""

    return


def convert_dataset(
    dataset: List[List[Atoms]],
    type_map: List[str], suffix, dest_dir="./"
):
    """"""
    from GDPy.computation.utils import get_composition_from_atoms
    groups = {}
    for atoms in dataset:
        composition = get_composition_from_atoms(atoms)
        key = "".join([k+str(v) for k,v in composition])
        if key in groups:
            groups[key].append(atoms)
        else:
            groups[key] = [atoms]
        
    # --- dpdata conversion
    import dpdata
    dest_dir = pathlib.Path(dest_dir)
    train_set_dir = dest_dir/f"{suffix}"
    if not train_set_dir.exists():
        train_set_dir.mkdir()
        
    cum_batchsizes = 0 # number of batchsizes for training
    for name, frames in groups.items():
        print(f"{suffix} system {name} nframes {len(frames)}")
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

        # --- convert data
        write(train_set_dir/f"{name}-{suffix}.xyz", frames_)
        dsys = dpdata.MultiSystems.from_file(
            train_set_dir/f"{name}-{suffix}.xyz", fmt="quip/gap/xyz", 
            type_map = type_map
        )
        dsys.to_deepmd_npy(train_set_dir) # prec, set_size

    return

def convert_groups(
    names: List[str], groups: List[List[Atoms]], batchsizes: Union[List[int],int],
    type_map: List[str], suffix, dest_dir="./"
):
    """"""
    nsystems = len(groups)
    if isinstance(batchsizes, int):
        batchsizes = [batchsizes]*nsystems

    # --- dpdata conversion
    import dpdata
    dest_dir = pathlib.Path(dest_dir)
    train_set_dir = dest_dir/f"{suffix}"
    if not train_set_dir.exists():
        train_set_dir.mkdir()
        
    cum_batchsizes = 0 # number of batchsizes for training
    for name, frames, batchsize in zip(names, groups, batchsizes):
        nframes = len(frames)
        nbatch = int(nframes / batchsize)
        print(f"{suffix} system {name} nframes {nframes} nbatch {nbatch}")
        cum_batchsizes += nbatch
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

        # --- convert data
        write(train_set_dir/f"{name}-{suffix}.xyz", frames_)
        dsys = dpdata.MultiSystems.from_file(
            train_set_dir/f"{name}-{suffix}.xyz", fmt="quip/gap/xyz", 
            type_map = type_map
        )
        dsys.to_deepmd_npy(train_set_dir) # prec, set_size

    return cum_batchsizes


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

        Args:
            dataset: [set_names, train_frames, test_frames]

        """
        # - add dataset to config
        if not dataset: # NOTE: can be a path or a List[Atoms]
            dataset = self.train_dataset
        assert dataset, f"No dataset has been set for the potential {self.name}."

        # - convert dataset
        #_ = convert_dataset(
        #    dataset[0], self.train_config["model"]["type_map"], "train", train_dir
        #)
        #_ = convert_dataset(
        #    dataset[1], self.train_config["model"]["type_map"], "valid", train_dir
        #)
        batchsizes = self.train_batchsize
        cum_batchsizes = convert_groups(
            dataset[0], dataset[1], batchsizes, 
            self.train_config["model"]["type_map"],
            "train", train_dir
        )
        _ = convert_groups(
            dataset[0], dataset[2], batchsizes,
            self.train_config["model"]["type_map"], 
            "valid", train_dir
        )

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