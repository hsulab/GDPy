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

from . import AbstractPotentialManager, AbstractTrainer
from . import DummyCalculator, CommitteeCalculator


class DeepmdDataloader():

    #: Datasset name.
    name: str = "deepmd"

    def __init__(
        self, batchsizes, cum_batchsizes, train_sys_dirs, valid_sys_dirs, 
        *args, **kwargs
    ) -> None:
        """"""
        self.batchsizes = batchsizes
        self.cum_batchsizes = cum_batchsizes
        self.train_sys_dirs = [str(x) for x in train_sys_dirs]
        self.valid_sys_dirs = [str(x) for x in valid_sys_dirs]

        return
    
    def as_dict(self, ) -> dict:
        """"""
        params = {}
        params["name"] = self.name
        params["batchsizes"] = self.batchsizes
        params["cum_batchsizes"] = self.cum_batchsizes
        params["train_sys_dirs"] = self.train_sys_dirs
        params["valid_sys_dirs"] = self.valid_sys_dirs

        return params


class DeepmdTrainer(AbstractTrainer):

    name = "deepmd"
    command = "dp"
    freeze_command = "dp"
    prefix = "config"

    #: Flag indicates that the training is finished properly.
    CONVERGENCE_FLAG: str = "finished training"

    def __init__(
        self, config: dict, type_list: List[str]=None, train_epochs: int=200,
        print_epochs: int = 5,
        directory=".", command="dp", freeze_command="dp", random_seed=1112, 
        *args, **kwargs
    ) -> None:
        """"""
        super().__init__(
            config=config, type_list=type_list, train_epochs=train_epochs,
            print_epochs=print_epochs,
            directory=directory, command=command, freeze_command=freeze_command, 
            random_seed=random_seed, *args, **kwargs
        )

        # - TODO: sync type_list
        if type_list is None:
            self._type_list = config["model"]["type_map"]
        else:
            self._type_list = type_list

        return

    def _resolve_train_command(self, init_model=None):
        """"""
        train_command = self.command

        # - add options
        command = "{} train {}.json ".format(train_command, self.name)
        if init_model is not None:
            init_model_path = pathlib.Path(init_model).resolve()
            if init_model_path.name.endswith(".pb"):
                command += "--init-frz-model {}".format(str(init_model_path))
            elif init_model_path.name.endswith("model.ckpt"):
                command += "--init-model {}".format(str(init_model_path))
            else:
                raise RuntimeError(f"Unknown init_model {str(init_model_path)}.")
        command += " 2>&1 > {}.out".format(self.name)

        return command

    def _resolve_freeze_command(self, *args, **kwargs):
        """"""
        freeze_command = self.command

        # - add options
        command = "{} freeze -o {} 2>&1 >> {}.out".format(
            freeze_command, self.frozen_name, self.name
        )

        return command

    def _resolve_compress_command(self, *args, **kwargs):
        """"""
        compress_command = self.command

        # - add options
        command = "{} compress -i {} -o {} 2>&1 >> {}.out".format(
            compress_command, self.frozen_name, f"{self.name}-c.pb", self.name
        )

        return command

    @property
    def frozen_name(self):
        """"""
        return f"{self.name}.pb"
    
    def _train_from_the_restart(self, dataset, init_model):
        """Train from the restart"""
        if not self.directory.exists():
            command = self._train_from_the_scratch(dataset, init_model)
        else:
            ckpt_info = self.directory/"checkpoint"
            if ckpt_info.exists() and ckpt_info.stat().st_size != 0:
                # TODO: check if the ckpt model exists?
                command = f"{self.command} train {self.name}.json "
                command += f"--restart model.ckpt"
                self._print(f"TRAINING COMMAND: {command}")
            else: # assume not at any ckpt so start from the scratch
                command = self._train_from_the_scratch(dataset, init_model)

        return command

    def _prepare_dataset(self, dataset, reduce_system: bool=False, *args, **kwargs):
        """"""
        if not self.directory.exists():
            self.directory.mkdir(parents=True, exist_ok=True)
        if not isinstance(dataset, DeepmdDataloader):
            set_names, train_frames, test_frames, adjusted_batchsizes = self._get_dataset(
                dataset, reduce_system
            )

            train_dir = self.directory

            # - update config
            self._print("--- write dp train data---")
            batchsizes = adjusted_batchsizes
            cum_batchsizes, train_sys_dirs = convert_groups(
                set_names, train_frames, batchsizes, 
                self.config["model"]["type_map"],
                "train", train_dir, self._print
            )
            _, valid_sys_dirs = convert_groups(
                set_names, test_frames, batchsizes,
                self.config["model"]["type_map"], 
                "valid", train_dir, self._print
            )
            self._print(f"accumulated number of batches: {cum_batchsizes}")

            dataset = DeepmdDataloader(
                batchsizes, cum_batchsizes, train_sys_dirs, valid_sys_dirs
            )
        else:
            ...

        return dataset

    def write_input(self, dataset, reduce_system: bool=False):
        """Write inputs for training.

        Args:
            reduce_system: Whether merge structures.

        """
        # - prepare dataset (convert dataset to DeepmdDataloader)
        dataset = self._prepare_dataset(dataset, reduce_system)

        # - check train config
        # NOTE: parameters
        #       numb_steps, seed
        #       descriptor-seed, fitting_net-seed
        #       training - training_data, validation_data
        train_config = copy.deepcopy(self.config)

        train_config["model"]["descriptor"]["seed"] =  self.rng.integers(0,10000, dtype=int)
        train_config["model"]["fitting_net"]["seed"] = self.rng.integers(0,10000, dtype=int)

        train_config["training"]["training_data"]["systems"] = [x for x in dataset.train_sys_dirs]
        train_config["training"]["training_data"]["batch_size"] = dataset.batchsizes

        train_config["training"]["validation_data"]["systems"] = [x for x in dataset.valid_sys_dirs]
        train_config["training"]["validation_data"]["batch_size"] = dataset.batchsizes

        train_config["training"]["seed"] = self.rng.integers(0,10000, dtype=int)

        # --- calc numb_steps
        min_freq_unit = 100.
        save_freq = int(np.ceil(dataset.cum_batchsizes*self.print_epochs/min_freq_unit)*min_freq_unit)
        train_config["training"]["save_freq"] = save_freq

        numb_steps = dataset.cum_batchsizes*self.train_epochs
        n_checkpoints = int(np.ceil(dataset.cum_batchsizes*self.train_epochs/save_freq))
        numb_steps = n_checkpoints*save_freq
        train_config["training"]["numb_steps"] = numb_steps

        # - write
        with open(self.directory/f"{self.name}.json", "w") as fopen:
            json.dump(train_config, fopen, indent=2)

        return

    def _get_dataset(self, dataset, reduce_system):
        """"""
        data_dirs = dataset.load()
        self._print("--- auto data reader ---")
        self._debug(data_dirs)

        batchsizes = dataset.batchsize
        nsystems = len(data_dirs)
        if isinstance(batchsizes, int):
            batchsizes = [batchsizes]*nsystems
        assert len(batchsizes) == nsystems, "Number of systems and batchsizes are inconsistent."

        # read configurations
        set_names = []
        train_size, test_size = [], []
        train_frames, test_frames = [], []
        adjusted_batchsizes = [] # auto-adjust batchsize based on nframes
        for i, (curr_system, curr_batchsize) in enumerate(zip(data_dirs, batchsizes)):
            curr_system = pathlib.Path(curr_system)
            set_name = "+".join(str(curr_system.relative_to(dataset.directory)).split("/"))
            set_names.append(set_name)
            self._print(f"System {set_name} Batchsize {curr_batchsize}")
            frames = [] # all frames in this subsystem
            subsystems = list(curr_system.glob("*.xyz"))
            subsystems.sort() # sort by alphabet
            for p in subsystems:
                # read and split dataset
                p_frames = read(p, ":")
                p_nframes = len(p_frames)
                frames.extend(p_frames)
                self._print(f"  subsystem: {p.name} number {p_nframes}")

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
                    ntrain = int(np.floor(nframes * dataset.train_ratio / new_batchsize) * new_batchsize)
                    train_index = self.rng.choice(nframes, ntrain, replace=False)
                    test_index = [x for x in range(nframes) if x not in train_index]
                adjusted_batchsizes.append(new_batchsize)

            ntrain, ntest = len(train_index), len(test_index)
            train_size.append(ntrain)
            test_size.append(ntest)

            self._print(f"    ntrain: {ntrain} ntest: {ntest} ntotal: {nframes} batchsize: {new_batchsize}")

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
            self._print(f"  Current Dataset -> ntrain: {n_train_frames} ntest: {n_test_frames}")

        assert len(train_size) == len(test_size), "inconsistent train_size and test_size"
        train_size = sum(train_size)
        test_size = sum(test_size)
        self._print(f"Total Dataset -> ntrain: {train_size} ntest: {test_size}")

        return set_names, train_frames, test_frames, adjusted_batchsizes

    def freeze(self):
        """"""
        # - freeze model
        frozen_model = super().freeze()

        # - compress model
        compressed_model = (self.directory/f"{self.name}-c.pb").absolute()
        if frozen_model.exists() and not compressed_model.exists():
            command = self._resolve_compress_command()
            try:
                proc = subprocess.Popen(command, shell=True, cwd=self.directory)
            except OSError as err:
                msg = "Failed to execute `{}`".format(command)
                # raise RuntimeError(msg) from err
                # self._print(msg)
                self._print("Failed to compress model.")
            except RuntimeError as err:
                self._print("Failed to compress model.")

            errorcode = proc.wait()
            if errorcode:
                path = os.path.abspath(self.directory)
                msg = ('Trainer "{}" failed with command "{}" failed in '
                       '{} with error code {}'.format(self.name, command,
                                                      path, errorcode))
                # NOTE: sometimes dp cannot compress the model
                #       this happens when the descriptor trainable is set False?
                # raise RuntimeError(msg)
                # self._print(msg)
                compressed_model.symlink_to(frozen_model.relative_to(compressed_model.parent))
        else:
            ...

        return compressed_model

    def read_convergence(self) -> bool:
        """Read training convergence.

        Check deepmd training progress by comparing the `numb_steps` in the input
        configuration and the current step in `lcurve.out`.
        
        """
        self._print(f"check {self.name} training convergence...")
        converged = False

        dpconfig_path = self.directory / f"{self.name}.json"
        if dpconfig_path.exists():
            # - get numb_steps
            with open(dpconfig_path, "r") as fopen:
                input_json = json.load(fopen)
            numb_steps = input_json["training"]["numb_steps"]

            # - get current step
            lcurve_out = self.directory / f"lcurve.out"
            if lcurve_out.exists():
                with open(lcurve_out, "r") as fopen:
                    lines = fopen.readlines()
                try:
                    curr_steps = int(lines[-1].strip().split()[0])
                    if curr_steps >= numb_steps:
                        converged = True
                    self._debug(f"{curr_steps} >=? {numb_steps}")
                except:
                    self._print(f"The endline of `lcure.out` is strange.")
            else:
                ...
        else:
            ...

        return converged


def convert_groups(
    names: List[str], groups: List[List[Atoms]], batchsizes: Union[List[int],int],
    type_map: List[str], suffix, dest_dir="./", pfunc=print
):
    """"""
    nsystems = len(groups)
    if isinstance(batchsizes, int):
        batchsizes = [batchsizes]*nsystems

    from gdpx.computation.utils import get_formula_from_atoms

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
        pfunc(f"{suffix} system {name} nframes {nframes} nbatch {nbatch} batchsize {batchsize}")
        # --- check composition consistent
        compositions = [get_formula_from_atoms(a) for a in frames]
        assert len(set(compositions)) == 1, f"Inconsistent composition {len(set(compositions))} =? 1..."
        curr_composition = compositions[0]

        cum_batchsizes += nbatch
        # --- NOTE: need convert forces to force
        frames_ = copy.deepcopy(frames) 
        # check pbc
        pbc = np.all([np.all(a.get_pbc()) for a in frames_])
        for atoms in frames_:
            try:
                forces = atoms.get_forces().copy()
                del atoms.arrays["forces"]
                atoms.arrays["force"] = forces
                keys = copy.deepcopy(list(atoms.arrays.keys()))
                for k in keys:
                    if k not in ["numbers", "positions", "force"]:
                        del atoms.arrays[k]
                #if "tags" in atoms.arrays:
                #    del atoms.arrays["tags"]
                #if "momenta" in atoms.arrays:
                #    del atoms.arrays["momenta"]
                #if "initial_charges" in atoms.arrays:
                #    del atoms.arrays["initial_charges"]
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
            if not pbc:
                with open(sys_dir/"nopbc", "w") as fopen:
                    fopen.write("nopbc\n")
        sys_dirs.append(sys_dir)
    (train_set_dir/"_temp").rmdir()

    return cum_batchsizes, sys_dirs


class DeepmdManager(AbstractPotentialManager):

    name = "deepmd"

    implemented_backends = ["ase", "lammps"]

    valid_combinations = (
        # calculator, dynamics
        ("ase", "ase"),
        ("lammps", "ase"),
        ("lammps", "lammps")
    )

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

        # TODO: make this a dataclass??
        #       currently, default disable uncertainty estimation
        estimate_uncertainty = calc_params.get("estimate_uncertainty", False)

        # - create specific calculator
        calc = DummyCalculator()
        if self.calc_backend == "ase":
            # return ase calculator
            try:
                #from deepmd.calculator import DP
                from gdpx.computation.dpx import DP
            except:
                raise ModuleNotFoundError("Please install deepmd-kit to use the ase interface.")
            #if models and type_map:
            #    calc = DP(model=models[0], type_dict=type_map)
            calcs = []
            for m in models:
                curr_calc = DP(model=m, type_dict=type_map)
                calcs.append(curr_calc)
            if len(calcs) == 1:
                calc = calcs[0]
            elif len(calcs) > 1:
                if estimate_uncertainty:
                    calc = CommitteeCalculator(calcs=calcs)
                else:
                    calc = calcs[0]
            else:
                ...
        elif self.calc_backend == "lammps":
            from gdpx.computation.lammps import Lammps
            if models:
                if len(models) == 1:
                    pair_style = "deepmd {}".format(" ".join(models))
                else:
                    if estimate_uncertainty:
                        pair_style = "deepmd {}".format(" ".join(models))
                    else:
                        pair_style = "deepmd {}".format(models[0])
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

        return calc

    def register_calculator(self, calc_params, *args, **kwargs) -> None:
        """ generate calculator with various backends
        """
        super().register_calculator(calc_params)
        
        self.calc = self._create_calculator(self.calc_params)

        return
    
    def switch_backend(self, backend: str=None) -> None:
        """Switch the potential's calculation backend."""
        if backend is None:
            return

        if not hasattr(self, "calc"):
            raise RuntimeError(f"{self.name} cannot switch backend as it does not have a calculator attached.")
        if backend not in self.implemented_backends:
            raise RuntimeError(f"{self.name} cannot switch backend from {self.calc_backend} to {backend}.")
        
        prev_backend = self.calc_backend
        if prev_backend == "ase" and backend == "lammps":
            calc_params = copy.deepcopy(self.calc_params)
            calc_params["backend"] = "lammps"
            command = calc_params.get("command", None)
            if command is None:
                raise RuntimeError(f"{self.name} cannot switch backend from ase to lammps as no command is provided.")
            self.register_calculator(calc_params)
        elif prev_backend == "lammps" and backend == "ase":
            calc_params = copy.deepcopy(self.calc_params)
            calc_params["backend"] = "ase"
            self.register_calculator(calc_params)
        else: # Nothing to do for other combinations
            ...

        return
    
    def switch_uncertainty_estimation(self, status: bool=True):
        """Switch on/off the uncertainty estimation."""
        # NOTE: Sometimes the manager loads several models and supports uncertainty
        #       by committee but the user disables it. We need change the calc to 
        #       the correct one as the loaded one is just a single calculator.
        if not hasattr(self, "calc"):
            raise RuntimeError("Fail to switch uncertainty status as it does not have a calc.")
        #print(f"{self.calc}")

        # NOTE: make sure manager.as_dict() can have correct param
        self.calc_params["estimate_uncertainty"] = status

        # - convert calculator
        if self.calc_backend == "ase":
            if status:
                if isinstance(self.calc, CommitteeCalculator):
                    ... # nothing to do
                else: # reload models
                    self.calc = self._create_calculator(self.calc_params)
            else:
                if isinstance(self.calc, CommitteeCalculator):
                    # TODO: save previous calc?
                    self.calc = self.calc.calcs[0]
                else:
                    ...
        elif self.calc_backend == "lammps":
            models = self.calc.pair_style.split()[1:] # model paths
            nmodels = len(models)
            if status:
                if nmodels > 1:
                    ...
                else:
                    self.calc = self._create_calculator(self.calc_params)
            else:
                # TODO: use self.calc_params? It should be protected?
                # pair_style deepmd m0 m1 m2 m3
                if nmodels > 1:
                    self.calc.pair_style = f"deepmd {models[0]}"
                else:
                    ...
        else:
            # TODO:
            # Other backends cannot have uncertainty estimation,
            # give a warning?
            ...
        #print(f"{self.calc}")

        return
    
    def remove_loaded_models(self, *args, **kwargs):
        """Loaded TF models should be removed before any copy.deepcopy operations."""
        self.calc.reset()
        if self.calc_backend == "ase":
            if isinstance(self.calc, CommitteeCalculator):
                for c in self.calc.calcs:
                    c.dp = None
            else:
                self.calc.dp = None
        else:
            ...

        return


if __name__ == "__main__":
    ...
