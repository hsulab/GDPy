#!/usr/bin/env python3
# -*- coding: utf-8 -*


import copy
import os
import pathlib

import yaml
from ase.io import read, write

from .calculators.dummy import DummyCalculator
from .manager import BasePotentialManager
from .trainer import BasePotentialTrainer
from .utils import build_a_committee_calculator, canonicalise_input_models


class NequipTrainer(BasePotentialTrainer):

    name = "nequip"
    command = "nequip-train"
    freeze_command = "nequip-deploy"
    prefix = "config"

    #: Training directory.
    RUN_NAME: str = "auto"

    def __init__(
        self,
        config: dict,
        type_list: list[str],
        train_epochs: int = 200,
        directory=".",
        command: str = "nequip-train",
        freeze_command: str = "nequip-deploy",
        random_seed: int = 1112,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(
            config=config,
            type_list=type_list,
            train_epochs=train_epochs,
            directory=directory,
            command=command,
            freeze_command=freeze_command,
            random_seed=random_seed,
            *args,
            **kwargs,
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
            # command += "--init-model {}".format(str(pathlib.Path(init_model).resolve()))
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
        for _, curr_system in enumerate(data_dirs):
            curr_system = pathlib.Path(curr_system)
            self._print(f"System {curr_system.stem}\n")
            curr_frames = []
            subsystems = list(curr_system.glob("*.xyz"))
            subsystems.sort()  # sort by alphabet
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

        write(self.directory / "dataset.xyz", frames)

        n_train = int(nframes * dataset.train_ratio / dataset.batchsize) * dataset.batchsize
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
        train_config["dataset_file_name"] = str((self.directory / "dataset.xyz").resolve())

        train_config["chemical_symbols"] = self.type_list

        train_config["batch_size"] = dataset.batchsize

        train_config["n_train"] = n_train
        train_config["n_val"] = n_val

        train_config["max_epochs"] = self.train_epochs

        with open(self.directory / f"{self.name}.yaml", "w") as fopen:
            yaml.safe_dump(train_config, fopen)

        return

    def read_convergence(self) -> bool:
        """"""
        converged = False
        with open(self.directory / self.RUN_NAME / "log", "rb") as fopen:
            try:  # catch OSError in case of a one line file
                fopen.seek(-2, os.SEEK_END)
                while fopen.read(1) != b"\n":
                    fopen.seek(-2, os.SEEK_CUR)
            except OSError:
                fopen.seek(0)
            line = fopen.readline().decode()

        if line.strip().startswith("Cumulative wall time"):
            converged = True

        return converged


class NequipManager(BasePotentialManager):

    name = "nequip"
    implemented_backends = ("ase", "lammps")

    valid_combinations = (
        ("ase", "ase"),
        ("lammps", "ase"),
        ("lammps", "lammps"),
    )

    def register_calculator(self, calc_params, *args, **kwargs):
        """Register the calculator."""
        super().register_calculator(calc_params, *args, **kwargs)

        calc_params = copy.deepcopy(calc_params)

        type_list = calc_params.pop("type_list", [])

        type_map = {}
        for i, a in enumerate(type_list):
            type_map[a] = i

        # Check if all models exist and update the self.calc_params
        # as the potential may be used in other directories if submitted by a scheduler.
        models = canonicalise_input_models(calc_params.pop("model", []))
        self.calc_params.update(model=models)

        estimate_uncertainty = calc_params.get("estimate_uncertainty", False)

        calc = DummyCalculator()
        if self.calc_backend == "ase":
            try:
                import torch
                from nequip.ase import NequIPCalculator  # type: ignore
            except:
                raise ModuleNotFoundError("Please install nequip and torch to use the ase interface.")
            device = "cuda" if torch.cuda.is_available() else "cpu"

            shared_params = dict(species_to_type_name={k: k for k in type_list}, device=device)
            params_list = []
            for m in models:
                specific_params = copy.deepcopy(shared_params)
                specific_params["model_path"] = m
                params_list.append(specific_params)
            num_models = len(models)
            if num_models > 0:
                calc = build_a_committee_calculator(
                    NequIPCalculator.from_deployed_model, params_list, estimate_uncertainty=estimate_uncertainty
                )
        elif self.calc_backend == "lammps":
            from gdpx.computation.lammps import Lammps

            command = calc_params.pop("command", None)

            flavour = calc_params.pop("flavour", "nequip")  # nequip or allegro
            if models:
                pair_style = f"{flavour}"
                pair_coeff = f"* * {str(models[0])}" + " {type_list}"
                calc = Lammps(
                    command=command,
                    pair_style=pair_style,
                    pair_coeff=pair_coeff,
                    **calc_params,
                )
                # Update several extra parameters
                calc.set(units="metal", atom_style="atomic")
                if pair_style == "nequip":
                    calc.set(newton="off")
                elif pair_style == "allegro":
                    calc.set(newton="on")
                else:
                    raise Exception(f"Unknown flavour {flavour} that must be nequip or allegro.")
        else:
            ...

        return


if __name__ == "__main__":
    ...
