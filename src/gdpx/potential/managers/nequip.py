#!/usr/bin/env python3
# -*- coding: utf-8 -*

import copy
import os
import pathlib
from typing import List
import warnings

import yaml

import numpy as np

from ase.io import read, write
from ase.calculators.calculator import Calculator

from . import AbstractPotentialManager, AbstractTrainer
from . import DummyCalculator, CommitteeCalculator


class NequipTrainer(AbstractTrainer):

    name = "nequip"
    command = "nequip-train"
    freeze_command = "nequip-deploy"
    prefix = "config"

    #: Training directory.
    RUN_NAME: str = "auto"

    def __init__(
        self, config: dict, type_list: List[str], train_epochs: int=200,
        directory=".", command: str="nequip-train", freeze_command: str="nequip-deploy",
        random_seed: int=1112, *args, **kwargs
    ) -> None:
        super().__init__(
            config=config, type_list=type_list, train_epochs=train_epochs,
            directory=directory, command=command, freeze_command=freeze_command, 
            random_seed=random_seed, *args, **kwargs
        )

        # - TODO: sync type_list
        self._type_list = type_list

        return

    def _resolve_train_command(self, init_model=None) -> str:
        """"""
        train_command = self.command

        # - add options
        command = "{} {}.yaml ".format(train_command, self.name)
        if init_model is not None:
            #command += "--init-model {}".format(str(pathlib.Path(init_model).resolve()))
            raise RuntimeError(f"{self.__class__.__name__} does not support init_model.")
        command += " 2>&1 > {}.out\n".format(self.name)

        return command

    def _resolve_freeze_command(self, *args, **kwargs) -> str:
        """"""
        freeze_command = self.freeze_command

        # - add options
        command = "{} build --train-dir {} {} 2>&1 >> {}.out".format(
            freeze_command, self.RUN_NAME, self.frozen_name, self.name
        )

        return command
    
    @property
    def frozen_name(self):
        """"""
        return f"{self.name}.pth"
    
    def write_input(self, dataset, *args, **kwargs):
        """"""
        # - check dataset
        data_dirs = dataset.load()
        self._print(data_dirs)
        self._print("--- auto data reader ---")

        frames = []
        for i, curr_system in enumerate(data_dirs):
            curr_system = pathlib.Path(curr_system)
            self._print(f"System {curr_system.stem}\n")
            curr_frames = []
            subsystems = list(curr_system.glob("*.xyz"))
            subsystems.sort() # sort by alphabet
            for p in subsystems:
                # read and split dataset
                p_frames = read(p, ":")
                p_nframes = len(p_frames)
                curr_frames.extend(p_frames)
                self._print(f"  subsystem: {p.name} number {p_nframes}")
            self._print(f"  nframes {len(curr_frames)}")
            frames.extend(curr_frames)
        nframes = len(frames)
        self._print(f"nframes {nframes}")

        write(self.directory/"dataset.xyz", frames)

        n_train = int(nframes*dataset.train_ratio/dataset.batchsize)*dataset.batchsize
        n_val = nframes - n_train

        # - check train config
        # params: root, run_name, seed, dataset_seed, n_train, n_val, batch_size
        #         dataset, dataset_file_name
        train_config = copy.deepcopy(self.config)

        train_config["root"] = str(self.directory.resolve())
        train_config["run_name"] = self.RUN_NAME

        train_config["seed"] = self.rng.integers(0, 10000, dtype=int)
        train_config["dataset_seed"] = self.rng.integers(0, 10000, dtype=int)

        train_config["dataset"] = "ase"
        train_config["dataset_file_name"] = str((self.directory/"dataset.xyz").resolve())

        train_config["chemical_symbols"] = self.type_list

        train_config["batch_size"] = dataset.batchsize

        train_config["n_train"] = n_train
        train_config["n_val"] = n_val

        train_config["max_epochs"] = self.train_epochs

        with open(self.directory/f"{self.name}.yaml", "w") as fopen:
            yaml.safe_dump(train_config, fopen)

        return
    
    def read_convergence(self) -> bool:
        """"""
        converged = False
        with open(self.directory/self.RUN_NAME/"log", "rb") as fopen:
            try:  # catch OSError in case of a one line file 
                fopen.seek(-2, os.SEEK_END)
                while fopen.read(1) != b'\n':
                    fopen.seek(-2, os.SEEK_CUR)
            except OSError:
                fopen.seek(0)
            line = fopen.readline().decode()
        
        if line.strip().startswith("Cumulative wall time"):
            converged = True

        return converged


class NequipManager(AbstractPotentialManager):

    name = "nequip"
    implemented_backends = ["ase", "lammps"]

    valid_combinations = (
        ("ase", "ase"), # calculator, dynamics
        ("lammps", "ase"),
        ("lammps", "lammps")
    )
    
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
        directory = calc_params.pop("directory", pathlib.Path.cwd())
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
                from nequip.ase import NequIPCalculator
            except:
                raise ModuleNotFoundError("Please install nequip and torch to use the ase interface.")
            calcs = []
            for m in models:
                curr_calc = NequIPCalculator.from_deployed_model(
                    model_path=m, species_to_type_name={k:k for k in atypes},
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
            from gdpx.computation.lammps import Lammps
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

        return calc

    def register_calculator(self, calc_params):
        """"""
        super().register_calculator(calc_params)
        
        self.calc = self._create_calculator(calc_params)

        return


if __name__ == "__main__":
    ...