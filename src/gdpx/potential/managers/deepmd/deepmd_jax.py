#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import copy
import json
import pathlib
from typing import Optional, List

import numpy as np

from .. import AbstractPotentialManager, AbstractTrainer, DummyCalculator

from .convert import convert_groups
from .deepmd import DeepmdDataloader


def get_model_fpaths(params: dict) -> List[str]:
    """Get model file paths from a dict.

    The file paths will be converted to be absolute.
    """
    model_ = params.get("model", [])
    if not isinstance(model_, list):
        model_ = [model_]

    models = []
    for m in model_:
        m = pathlib.Path(m).resolve()
        if not m.exists():
            raise FileNotFoundError(f"Cant find model file {str(m)}")
        models.append(str(m))

    return models


class DeepmdJaxManager(AbstractPotentialManager):

    name: str = "deepmd_jax"

    implemented_backends: List[str] = ["ase", "jax"]

    valid_combinations = (
        ("ase", "ase"),
        ("jax", "jax"),
    )

    def register_calculator(self, calc_params: dict, *args, **kwargs) -> None:
        """generate calculator with various backends"""
        super().register_calculator(calc_params)
        calc_params = copy.deepcopy(calc_params)

        type_list = calc_params.pop("type_list", [])
        type_map = {}
        for i, a in enumerate(type_list):
            type_map[a] = i

        models = get_model_fpaths(calc_params)
        self.calc_params.update(model=models)

        calc = DummyCalculator()
        if self.calc_backend == "ase" or self.calc_backend == "jax":
            try:
                from .dpjax import DPJax
            except:
                raise ModuleNotFoundError(
                    "Please install deepmd-jax to use the jax interface."
                )
            # TODO: only support one model...
            if models:
                calc = DPJax(
                    model=models[0],
                    type_map=type_map,
                )
                print(f"{calc =}")
            else:
                ...  # No models.
        else:
            ...

        self.calc = calc

        return


class DeepmdJaxTrainer(AbstractTrainer):

    name = "deepmd_jax"

    def __init__(self, type_list: Optional[List[str]]=None, *args, **kwargs):
        """"""
        super().__init__(type_list=type_list, *args, **kwargs)

        if type_list is None:
            ...
        else:
            self._type_list = type_list

        assert sorted(self.type_list) == self.type_list, f"DeepmdJaxTrainer must have a type list in the alphabetical order."

        return

    def _resolve_train_command(self, *args, **kwars):
        """"""

        return

    def _resolve_freeze_command(self, *args, **kwargs):
        """"""

        return

    @property
    def frozen_name(self) -> str:
        """"""

        return f"{self.name}.pkl"

    def write_input(self, dataset):
        """Write inputs for training."""
        dataset = self._prepare_dataset(dataset)

        train_config = copy.deepcopy(self.config)

        train_config["model_type"] = "energy"
        train_config["save_path"] = str(self.directory/self.frozen_name)
        train_config["train_data_path"] = [
            [x] for x in dataset.train_sys_dirs
        ]
        train_config["val_data_path"] = [
            [x] for x in dataset.valid_sys_dirs
        ]

        if isinstance(dataset.batchsize, int):
            train_config["batch_size"] = dataset.batchsize
        elif isinstance(dataset.batchsize, str):
            method, new_batchsize = dataset.batchsize.split(":")
            if method == "n_atoms":
                train_config["label_bs"] = int(new_batchsize)
            elif method == "n_structures":
                train_config["batch_size"] = int(new_batchsize)
            else:
                raise RuntimeError(f"Unknown batchszie `{dataset.batchsize}`.")
        else:
            raise RuntimeError(f"Unknown batchszie `{dataset.batchsize}`.")

        min_freq_unit = 100.0
        save_freq = int(
            np.ceil(dataset.cum_batchsizes * self.print_epochs / min_freq_unit)
            * min_freq_unit
        )
        train_config["print_every"] = save_freq

        numb_steps = dataset.cum_batchsizes * self.train_epochs
        n_checkpoints = int(
            np.ceil(dataset.cum_batchsizes * self.train_epochs / save_freq)
        )
        numb_steps = n_checkpoints * save_freq
        train_config["step"] = numb_steps

        train_config["seed"] = self.rng.integers(0, 1e8, dtype=int)

        with open(self.directory/"deepmd_jax.json", "w") as fopen:
            json.dump(train_config, fopen, indent=2)

        return

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
                self.type_list,
                "train",
                train_dir,
                self._print,
            )
            _, valid_sys_dirs = convert_groups(
                set_names,
                test_frames,
                batchsizes,
                self.type_list,
                "valid",
                train_dir,
                self._print,
            )
            self._print(f"accumulated number of batches: {cum_batchsizes}")

            dataset = DeepmdDataloader(
                dataset.batchsize, batchsizes, cum_batchsizes, train_sys_dirs, valid_sys_dirs
            )
        else:
            ...

        return dataset


    def train(self, dataset, init_model=None, *args, **kwargs):
        """"""
        self._print("TRAINING INTERNALLY.")

        if not self.directory.exists():
            self.directory.mkdir(parents=True, exist_ok=True)
        self.write_input(dataset)

        with open(self.directory/"deepmd_jax.json", "r") as fopen:
            train_config = json.load(fopen)

        self._print(f"{train_config}")

        from deepmd_jax.train import train
        _ = train(**train_config)

        return

    def freeze(self):
        """No freeze and compress need done."""
        frozen_model = (self.directory / self.frozen_name).resolve()

        return frozen_model

    def read_convergence(self) -> bool:
        """"""
        converged = False

        if (self.directory/self.frozen_name):
            converged = True

        return converged


if __name__ == "__main__":
    ...
