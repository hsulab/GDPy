#!/usr/bin/env python3
# -*- coding: utf-8 -*

import os
import copy
import dataclasses
import pathlib
import subprocess
from typing import Optional, Union, List, Tuple, Callable

import json

import numpy as np

from ase import Atoms
from ase.io import read, write
from ase.calculators.calculator import Calculator
from ase.calculators.singlepoint import SinglePointCalculator

from .. import AbstractPotentialManager, AbstractTrainer
from .. import DummyCalculator, CommitteeCalculator

from .convert import convert_groups


@dataclasses.dataclass
class DeepmdSystem:

    #: System name.
    name: str

    #: Frames.
    frames: List[Atoms]

    #: Number of train_and_split.
    train_and_split: Tuple[int, int]

    #: Batchsize
    batchsize: int = 1

    def __post_init__(
        self,
    ):
        """"""
        self.nframes = len(self.frames)
        assert self.nframes == sum(
            self.train_and_split
        ), f"{self.name}: {self.nframes} != sum({self.train_and_split})"

        composition_list = [a.get_chemical_formula() for a in self.frames]
        assert len(set(composition_list)) == 1, f"{self.name}: {composition_list[0]}?"
        self.composition = composition_list[0]

        return

    def write(self, directory: pathlib.Path) -> None:
        """"""
        directory.mkdir(parents=True, exist_ok=True)
        write(directory / f"{self.composition}.xyz", self.frames)

        return


class DeepmdDataloader:

    #: Datasset name.
    name: str = "deepmd"

    #:
    _systems: List[DeepmdSystem] = []

    def __init__(
        self,
        batchsizes,
        cum_batchsizes,
        train_sys_dirs,
        valid_sys_dirs,
        *args,
        **kwargs,
    ) -> None:
        """"""
        self.batchsizes = batchsizes
        self.cum_batchsizes = cum_batchsizes
        self.train_sys_dirs = [str(x) for x in train_sys_dirs]
        self.valid_sys_dirs = [str(x) for x in valid_sys_dirs]

        return

    @staticmethod
    def from_directory(
        train_directory: Union[str, pathlib.Path],
        valid_directory: Optional[Union[str, pathlib.Path]]=None,
    ) -> "DeepmdDataloader":
        """"""
        # - read trainset...
        train_wdir = pathlib.Path(train_directory)
        trainset_dirs = sorted(train_wdir.iterdir())

        if valid_directory is not None:
            valid_wdir = pathlib.Path(valid_directory)
            validset_dirs = sorted(valid_wdir.iterdir())
        else:
            validset_dirs = []

        batchsizes = [1]*len(trainset_dirs)
        cum_batchsizes = sum(batchsizes)

        return DeepmdDataloader(batchsizes, cum_batchsizes, trainset_dirs, validset_dirs)

    @property
    def systems(
        self,
    ):
        """"""

        return self._systems

    def load(
        self,
    ) -> None:
        """"""
        systems = []
        for p in self.train_sys_dirs:
            p = pathlib.Path(p)
            print(p)
            curr_train_frames = DeepmdDataloader.convert_system_to_frames(p)
            num_curr_train_frames = len(curr_train_frames)
            curr_frames = curr_train_frames
            for p2 in self.valid_sys_dirs:
                p2 = pathlib.Path(p2)
                if p2.name == p.name:
                    curr_valid_frames = DeepmdDataloader.convert_system_to_frames(p2)
                    num_curr_valid_frames = len(curr_valid_frames)
                    curr_frames.extend(curr_valid_frames)
                    break
            else:
                num_curr_valid_frames = 0
            curr_system = DeepmdSystem(
                p.name, curr_frames, (num_curr_train_frames, num_curr_valid_frames)
            )
            systems.append(curr_system)
        self._systems = systems

        return

    def dump(self, directory: Union[str, pathlib.Path]) -> None:
        """"""
        directory = pathlib.Path(directory)
        for curr_system in self.systems:
            curr_system.write(
                directory / (curr_system.name + "-" + curr_system.composition),
            )
            print(
                f"{curr_system.name =} {curr_system.composition} {curr_system.train_and_split} {curr_system.nframes}"
            )

        return

    @staticmethod
    def set2frames(
        set_dir: pathlib.Path, chemical_symbols: List[str], pbc: List[int]
    ) -> List[Atoms]:
        """Convert set into frames."""
        boxes = np.load(set_dir / "box.npy")
        nframes = boxes.shape[0]
        coords = np.load(set_dir / "coord.npy")
        energies = np.load(set_dir / "energy.npy")
        forces = np.load(set_dir / "force.npy")

        if nframes == 1:
            energies = energies.reshape(-1)

        frames = []
        for i in range(nframes):
            cell = boxes[i, :].reshape(3, 3)
            positions = coords[i, :].reshape(-1, 3)
            atoms = Atoms(
                symbols=chemical_symbols, positions=positions, cell=cell, pbc=pbc
            )
            results = {"energy": energies[i], "forces": forces[i, :].reshape(-1, 3)}
            spc = SinglePointCalculator(atoms, **results)
            atoms.calc = spc
            frames.append(atoms)

        return frames

    @staticmethod
    def convert_system_to_frames(curr_system: pathlib.Path):
        """"""
        # find all set dirs
        set_dirs = sorted(curr_system.glob("set*"))

        type_list = np.loadtxt(curr_system / "type_map.raw", dtype=str)
        if len(type_list.shape) == 0:
            type_list = type_list.reshape(-1)
        atype = np.loadtxt(curr_system / "type.raw", dtype=int)
        if len(atype.shape) == 0:
            atype = atype.reshape(-1)
        chemical_symbols = [type_list[a] for a in atype]
        pbc = [1, 1, 1] if not (curr_system / "nopbc").exists() else [0, 0, 0]

        # train data
        frames = []
        for set_dir in set_dirs[:]:
            frames.extend(DeepmdDataloader.set2frames(set_dir, chemical_symbols, pbc))

        return frames

    def as_dict(
        self,
    ) -> dict:
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
        self,
        config: dict,
        type_list: List[str] = None,
        train_epochs: int = 200,
        print_epochs: int = 5,
        directory=".",
        command="dp",
        freeze_command="dp",
        random_seed=1112,
        *args,
        **kwargs,
    ) -> None:
        """"""
        super().__init__(
            config=config,
            type_list=type_list,
            train_epochs=train_epochs,
            print_epochs=print_epochs,
            directory=directory,
            command=command,
            freeze_command=freeze_command,
            random_seed=random_seed,
            *args,
            **kwargs,
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
            ckpt_info = self.directory / "checkpoint"
            if ckpt_info.exists() and ckpt_info.stat().st_size != 0:
                # TODO: check if the ckpt model exists?
                command = f"{self.command} train {self.name}.json "
                command += f"--restart model.ckpt"
                self._print(f"TRAINING COMMAND: {command}")
            else:  # assume not at any ckpt so start from the scratch
                command = self._train_from_the_scratch(dataset, init_model)

        return command

    def _prepare_dataset(self, dataset, *args, **kwargs):
        """"""
        if not self.directory.exists():
            self.directory.mkdir(parents=True, exist_ok=True)
        if not isinstance(dataset, DeepmdDataloader):
            set_names, train_frames, test_frames, adjusted_batchsizes = (
                dataset.split_train_and_test()
            )
            train_dir = self.directory

            # - update config
            self._print("--- write dp train data---")
            batchsizes = adjusted_batchsizes
            cum_batchsizes, train_sys_dirs = convert_groups(
                set_names,
                train_frames,
                batchsizes,
                self.config["model"]["type_map"],
                "train",
                train_dir,
                self._print,
            )
            _, valid_sys_dirs = convert_groups(
                set_names,
                test_frames,
                batchsizes,
                self.config["model"]["type_map"],
                "valid",
                train_dir,
                self._print,
            )
            self._print(f"accumulated number of batches: {cum_batchsizes}")

            dataset = DeepmdDataloader(
                batchsizes, cum_batchsizes, train_sys_dirs, valid_sys_dirs
            )
        else:
            ...

        return dataset

    def write_input(self, dataset):
        """Write inputs for training."""
        # - prepare dataset (convert dataset to DeepmdDataloader)
        dataset = self._prepare_dataset(dataset)

        # - check train config
        # NOTE: parameters
        #       numb_steps, seed
        #       descriptor-seed, fitting_net-seed
        #       training - training_data, validation_data
        train_config = copy.deepcopy(self.config)

        train_config["model"]["descriptor"]["seed"] = self.rng.integers(
            0, 10000, dtype=int
        )
        train_config["model"]["fitting_net"]["seed"] = self.rng.integers(
            0, 10000, dtype=int
        )

        train_config["training"]["training_data"]["systems"] = [
            x for x in dataset.train_sys_dirs
        ]
        train_config["training"]["training_data"]["batch_size"] = dataset.batchsizes

        train_config["training"]["validation_data"]["systems"] = [
            x for x in dataset.valid_sys_dirs
        ]
        train_config["training"]["validation_data"]["batch_size"] = dataset.batchsizes

        train_config["training"]["seed"] = self.rng.integers(0, 10000, dtype=int)

        # --- calc numb_steps
        min_freq_unit = 100.0
        save_freq = int(
            np.ceil(dataset.cum_batchsizes * self.print_epochs / min_freq_unit)
            * min_freq_unit
        )
        train_config["training"]["save_freq"] = save_freq

        numb_steps = dataset.cum_batchsizes * self.train_epochs
        n_checkpoints = int(
            np.ceil(dataset.cum_batchsizes * self.train_epochs / save_freq)
        )
        numb_steps = n_checkpoints * save_freq
        train_config["training"]["numb_steps"] = numb_steps

        # - write
        with open(self.directory / f"{self.name}.json", "w") as fopen:
            json.dump(train_config, fopen, indent=2)

        return

    def freeze(self):
        """"""
        # - freeze model
        frozen_model = super().freeze()

        # - compress model
        compressed_model = (self.directory / f"{self.name}-c.pb").absolute()
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
                msg = (
                    'Trainer "{}" failed with command "{}" failed in '
                    "{} with error code {}".format(self.name, command, path, errorcode)
                )
                # NOTE: sometimes dp cannot compress the model
                #       this happens when the descriptor trainable is set False?
                # raise RuntimeError(msg)
                # self._print(msg)
                compressed_model.symlink_to(
                    frozen_model.relative_to(compressed_model.parent)
                )
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


class DeepmdManager(AbstractPotentialManager):

    name = "deepmd"

    implemented_backends = ["ase", "lammps"]

    valid_combinations = (
        # calculator, dynamics
        ("ase", "ase"),
        ("lammps", "ase"),
        ("lammps", "lammps"),
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
        directory = calc_params.pop("directory", pathlib.Path.cwd())

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
            m = pathlib.Path(m).resolve()
            if not m.exists():
                raise FileNotFoundError(f"Cant find model file {str(m)}")
            models.append(str(m))
        self.calc_params.update(model=models)

        # TODO: make this a dataclass??
        #       currently, default disable uncertainty estimation
        estimate_uncertainty = calc_params.get("estimate_uncertainty", False)

        # - create specific calculator
        calc = DummyCalculator()
        if self.calc_backend == "ase":
            # return ase calculator
            try:
                from .calculator import DP
            except:
                raise ModuleNotFoundError(
                    "Please install deepmd-kit to use the ase interface."
                )
            # if models and type_map:
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
                assert (
                    pair_style_name == "deepmd"
                ), "Incorrect pair_style for lammps deepmd..."

                calc = Lammps(
                    command=command,
                    directory=directory,
                    pair_style=pair_style,
                    pair_coeff=pair_coeff,
                    **calc_params,
                )
                # - update several params
                calc.units = "metal"
                calc.atom_style = "atomic"

        return calc

    def register_calculator(self, calc_params, *args, **kwargs) -> None:
        """generate calculator with various backends"""
        super().register_calculator(calc_params)

        self.calc = self._create_calculator(self.calc_params)

        return

    def switch_backend(self, backend: str = None) -> None:
        """Switch the potential's calculation backend."""
        if backend is None:
            return

        if not hasattr(self, "calc"):
            raise RuntimeError(
                f"{self.name} cannot switch backend as it does not have a calculator attached."
            )
        if backend not in self.implemented_backends:
            raise RuntimeError(
                f"{self.name} cannot switch backend from {self.calc_backend} to {backend}."
            )

        prev_backend = self.calc_backend
        if prev_backend == "ase" and backend == "lammps":
            calc_params = copy.deepcopy(self.calc_params)
            calc_params["backend"] = "lammps"
            command = calc_params.get("command", None)
            if command is None:
                raise RuntimeError(
                    f"{self.name} cannot switch backend from ase to lammps as no command is provided."
                )
            else:
                self.calc_backend = None
            self.register_calculator(calc_params)
        elif prev_backend == "lammps" and backend == "ase":
            calc_params = copy.deepcopy(self.calc_params)
            calc_params["backend"] = "ase"
            self.calc_backend = None
            self.register_calculator(calc_params)
        else:  # Nothing to do for other combinations
            ...

        return

    def switch_uncertainty_estimation(self, status: bool = True):
        """Switch on/off the uncertainty estimation."""
        # NOTE: Sometimes the manager loads several models and supports uncertainty
        #       by committee but the user disables it. We need change the calc to
        #       the correct one as the loaded one is just a single calculator.
        if not hasattr(self, "calc"):
            raise RuntimeError(
                "Fail to switch uncertainty status as it does not have a calc."
            )
        # print(f"{self.calc}")

        # NOTE: make sure manager.as_dict() can have correct param
        self.calc_params["estimate_uncertainty"] = status

        # - convert calculator
        if self.calc_backend == "ase":
            if status:
                if isinstance(self.calc, CommitteeCalculator):
                    ...  # nothing to do
                else:  # reload models
                    self.calc = self._create_calculator(self.calc_params)
            else:
                if isinstance(self.calc, CommitteeCalculator):
                    # TODO: save previous calc?
                    self.calc = self.calc.calcs[0]
                else:
                    ...
        elif self.calc_backend == "lammps":
            models = self.calc.pair_style.split()[1:]  # model paths
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
        # print(f"{self.calc}")

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
