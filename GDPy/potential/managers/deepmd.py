#!/usr/bin/env python3
# -*- coding: utf-8 -*

import os
import copy
from pathlib import Path
import pathlib
import subprocess
from typing import Union, List, Callable

import json

import numpy as np

from ase import Atoms
from ase.io import read, write
from ase.calculators.calculator import Calculator

from GDPy.core.register import registers
from GDPy.potential.manager import AbstractPotentialManager
from GDPy.potential.trainer import AbstractTrainer


@registers.trainer.register
class DeepmdTrainer(AbstractTrainer):

    name = "deepmd"
    command = "dp"
    prefix = "config"

    def __init__(
        self, config: dict, train_ratio: float=0.9, train_epochs: int=200,
        directory=".", command="dp", random_seed=1112, 
        *args, **kwargs
    ) -> None:
        """"""
        super().__init__(
            config=config, train_ratio=train_ratio, train_epochs=train_epochs,
            directory=directory, command=command, random_seed=random_seed, *args, **kwargs
        )

        # - update command
        if self.prefix not in self.command:
            train_command = self.command.strip().split()[0]
            self.command = "{} train {}.json".format(train_command, self.prefix)

        return
    
    def write_input(self, dataset, batchsizes, reduce_system: bool=False):
        """Write inputs for training.

        Args:
            reduce_system: Whether merge structures.

        """
        set_names, train_frames, test_frames, adjusted_batchsizes = self._get_dataset(
            dataset, batchsizes, reduce_system
        )

        train_dir = self.directory

        # - update config
        batchsizes = adjusted_batchsizes
        cum_batchsizes, train_sys_dirs = convert_groups(
            set_names, train_frames, batchsizes, 
            self.config["model"]["type_map"],
            "train", train_dir
        )
        _, valid_sys_dirs = convert_groups(
            set_names, test_frames, batchsizes,
            self.config["model"]["type_map"], 
            "valid", train_dir
        )

        # - check train config
        # NOTE: parameters
        #       numb_steps, seed
        #       descriptor-seed, fitting_net-seed
        #       training - training_data, validation_data
        train_config = copy.deepcopy(self.config)

        train_config["model"]["descriptor"]["seed"] =  self.rng.integers(0,10000, dtype=int)
        train_config["model"]["fitting_net"]["seed"] = self.rng.integers(0,10000, dtype=int)

        train_config["training"]["training_data"]["systems"] = [str(x.resolve()) for x in train_sys_dirs]
        train_config["training"]["training_data"]["batch_size"] = batchsizes

        train_config["training"]["validation_data"]["systems"] = [str(x.resolve()) for x in valid_sys_dirs]
        train_config["training"]["validation_data"]["batch_size"] = batchsizes

        train_config["training"]["seed"] = self.rng.integers(0,10000, dtype=int)

        # --- calc numb_steps
        save_freq = train_config["training"]["save_freq"]
        n_checkpoints = int(np.ceil(cum_batchsizes*self.train_epochs/save_freq))
        numb_steps = n_checkpoints*save_freq

        train_config["training"]["numb_steps"] = numb_steps

        # - write
        with open(train_dir/"config.json", "w") as fopen:
            json.dump(train_config, fopen, indent=2)

        return
    
    def _get_dataset(self, dataset, batchsizes, reduce_system):
        data_dirs = dataset.load()
        self._print(data_dirs)
        self._print("\n--- auto data reader ---\n")
        #content += f"Use batchsize {batchsize} and train-ratio {train_ratio}\n"

        nsystems = len(data_dirs)
        if isinstance(batchsizes, int):
            batchsizes = [batchsizes]*nsystems
        assert len(batchsizes) == nsystems, "Number of systems and batchsizes are inconsistent."

        # read configurations
        set_names = []
        train_size, test_size = [], []
        train_frames, test_frames = [], []
        adjusted_batchsizes = [] # auto-adjust batchsize based on nframes
        for i, (cur_system, curr_batchsize) in enumerate(zip(data_dirs, batchsizes)):
            cur_system = pathlib.Path(cur_system)
            set_names.append(cur_system.name)
            self._print(f"System {cur_system.stem} Batchsize {curr_batchsize}\n")
            frames = [] # all frames in this subsystem
            subsystems = list(cur_system.glob("*.xyz"))
            subsystems.sort() # sort by alphabet
            for p in subsystems:
                # read and split dataset
                p_frames = read(p, ":")
                p_nframes = len(p_frames)
                frames.extend(p_frames)
                self._print(f"  subsystem: {p.name} number {p_nframes}\n")

            # split dataset and get adjusted batchsize
            # TODO: adjust batchsize of train and test separately
            nframes = len(frames)
            if nframes <= curr_batchsize:
                if nframes == 1 or curr_batchsize == 1:
                    new_batchsize = 1
                else:
                    new_batchsize = int(2**np.floor(np.log2(nframes)))
                adjusted_batchsizes.append(new_batchsize)
                # NOTE: use same train and test set
                #       since they are very important structures...
                train_index = list(range(nframes))
                test_index = list(range(nframes))
            else:
                if nframes == 1 or curr_batchsize == 1:
                    new_batchsize = 1
                    train_index = list(range(nframes))
                    test_index = list(range(nframes))
                else:
                    new_batchsize = curr_batchsize
                    # - assure there is at least one batch for test
                    #          and number of train frames is integer times of batchsize
                    ntrain = int(np.floor(nframes * self.train_ratio / new_batchsize) * new_batchsize)
                    train_index = self.rng.choice(nframes, ntrain, replace=False)
                    test_index = [x for x in range(nframes) if x not in train_index]
                adjusted_batchsizes.append(new_batchsize)

            ntrain, ntest = len(train_index), len(test_index)
            train_size.append(ntrain)
            test_size.append(ntest)

            self._print(f"    ntrain: {ntrain} ntest: {ntest} ntotal: {nframes} batchsize: {new_batchsize}\n")

            curr_train_frames = [frames[train_i] for train_i in train_index]
            curr_test_frames = [frames[test_i] for test_i in test_index]
            if reduce_system:
                # train
                train_frames.extend(curr_train_frames)
                n_train_frames = len(train_frames)

                # test
                test_frames.extend(curr_test_frames)
                n_test_frames = len(test_frames)
            else:
                # train
                train_frames.append(curr_train_frames)
                n_train_frames = sum([len(x) for x in train_frames])

                # test
                test_frames.append(curr_test_frames)
                n_test_frames = sum([len(x) for x in test_frames])
            self._print(f"  Current Dataset -> ntrain: {n_train_frames} ntest: {n_test_frames}\n\n")

        assert len(train_size) == len(test_size), "inconsistent train_size and test_size"
        train_size = sum(train_size)
        test_size = sum(test_size)
        self._print(f"Total Dataset -> ntrain: {train_size} ntest: {test_size}\n")

        return set_names, train_frames, test_frames, adjusted_batchsizes
    
    def read_convergence(self):
        """"""
        CONVERGENCE_FLAG = "finished training"

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

    from GDPy.computation.utils import get_formula_from_atoms

    # --- dpdata conversion
    import dpdata
    dest_dir = pathlib.Path(dest_dir)
    train_set_dir = dest_dir/f"{suffix}"
    if not train_set_dir.exists():
        train_set_dir.mkdir()

    sys_dirs = []
        
    cum_batchsizes = 0 # number of batchsizes for training
    for name, frames, batchsize in zip(names, groups, batchsizes):
        nframes = len(frames)
        nbatch = int(np.ceil(nframes / batchsize))
        print(f"{suffix} system {name} nframes {nframes} nbatch {nbatch}")
        # --- check composition consistent
        compositions = [get_formula_from_atoms(a) for a in frames]
        assert len(set(compositions)) == 1, "Inconsistent composition..."
        curr_composition = compositions[0]

        cum_batchsizes += nbatch
        # --- NOTE: need convert forces to force
        frames_ = copy.deepcopy(frames) 
        for atoms in frames_:
            try:
                forces = atoms.get_forces().copy()
                del atoms.arrays["forces"]
                atoms.arrays["force"] = forces
                # TODO: dpdata doesnot support read tags
                if "tags" in atoms.arrays:
                    del atoms.arrays["tags"]
                if "initial_charges" in atoms.arrays:
                    del atoms.arrays["initial_charges"]
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
        # NOTE: this function create dir with composition and overwrite files
        #       so we need separate dirs...
        sys_dir = train_set_dir/name
        if sys_dir.exists():
            raise FileExistsError(f"{sys_dir} exists. Please check the dataset.")
        else:
            dsys.to_deepmd_npy(train_set_dir/"_temp") # prec, set_size
            (train_set_dir/"_temp"/curr_composition).rename(sys_dir)
        sys_dirs.append(sys_dir)
    (train_set_dir/"_temp").rmdir()

    return cum_batchsizes, sys_dirs


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
        batchsizes = dataset[3]
        cum_batchsizes, train_sys_dirs = convert_groups(
            dataset[0], dataset[1], batchsizes, 
            self.train_config["model"]["type_map"],
            "train", train_dir
        )
        _, valid_sys_dirs = convert_groups(
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

        train_config["training"]["training_data"]["systems"] = [str(x.resolve()) for x in train_sys_dirs]
        train_config["training"]["training_data"]["batch_size"] = batchsizes

        train_config["training"]["validation_data"]["systems"] = [str(x.resolve()) for x in valid_sys_dirs]
        train_config["training"]["validation_data"]["batch_size"] = batchsizes

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