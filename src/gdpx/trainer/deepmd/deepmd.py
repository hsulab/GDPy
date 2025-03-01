#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import copy
import json
import os
import pathlib
import subprocess
from typing import Optional

import numpy as np

from gdpx.dataloader.deepmd import DeepmdDataloader

from ..trainer import BasePotentialTrainer
from .convert import convert_groups


class DeepmdTrainer(BasePotentialTrainer):

    name = "deepmd"
    command = "dp"
    freeze_command = "dp"
    prefix = "config"

    #: Flag indicates that the training is finished properly.
    CONVERGENCE_FLAG: str = "finished training"

    def __init__(
        self,
        config: dict,
        type_list: Optional[list[str]] = None,
        train_epochs: int = 200,
        print_epochs: int = 5,
        directory=".",
        command: str = "dp",
        train_options: str = "",
        freeze_command: str = "dp",
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

        self.train_options = train_options

        return

    def _resolve_train_command(self, init_model=None):
        """"""
        train_command = self.command

        # - add options
        command = "{} train {}.json {} ".format(train_command, self.name, self.train_options)
        if init_model is not None:
            init_model_path = pathlib.Path(init_model).resolve()
            if init_model_path.name.endswith(".pb"):
                command += " --init-frz-model {}".format(str(init_model_path))
            elif init_model_path.name.endswith("model.ckpt"):
                command += " --init-model {}".format(str(init_model_path))
            else:
                raise RuntimeError(f"Unknown init_model {str(init_model_path)}.")
        command += " 2>&1 > {}.out".format(self.name)

        return command

    def _resolve_freeze_command(self, *args, **kwargs):
        """"""
        freeze_command = self.command

        # - add options
        command = "{} freeze -o {} 2>&1 >> {}.out".format(freeze_command, self.frozen_name, self.name)

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
            set_names, train_frames, test_frames, adjusted_batchsizes = dataset.split_train_and_test()
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
                dataset.batchsize,  # FIXME: xyz_dataset has this?
                batchsizes,
                cum_batchsizes,
                train_sys_dirs,
                valid_sys_dirs,
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

        train_config["model"]["descriptor"]["seed"] = self.rng.integers(0, 10000, dtype=int)
        train_config["model"]["fitting_net"]["seed"] = self.rng.integers(0, 10000, dtype=int)

        train_config["training"]["training_data"]["systems"] = [x for x in dataset.train_sys_dirs]
        train_config["training"]["training_data"]["batch_size"] = dataset.batchsizes

        # verify validation_data
        validation_data, validation_batchsizes = [], []
        for v_system, v_batchsize in zip(dataset.valid_sys_dirs, dataset.batchsizes):
            if v_system != "None":  # None will be saved to a string before
                validation_data.append(v_system)
                validation_batchsizes.append(v_batchsize)
        if validation_data:
            train_config["training"]["validation_data"]["systems"] = validation_data
            train_config["training"]["validation_data"]["batch_size"] = validation_batchsizes
        else:
            if "validation_data" in train_config["training"]:
                train_config["training"].pop("validation_data", None)

        train_config["training"]["seed"] = self.rng.integers(0, 10000, dtype=int)

        # Determine `numb_steps`
        min_freq_unit = 100.0
        save_freq = int(np.ceil(dataset.cum_batchsizes * self.print_epochs / min_freq_unit) * min_freq_unit)
        train_config["training"]["save_freq"] = save_freq

        # NOTE: Currently, we check whether the training is fininished by steps in lcurve.out.
        #       Thus, we need make sure the last step (numb_steps) is displayed in lcurve.out
        #       by making numb_steps can be divided by disp_freq.
        train_config["training"]["disp_freq"] = save_freq

        numb_steps = dataset.cum_batchsizes * self.train_epochs
        num_checkpoints = int(np.ceil(dataset.cum_batchsizes * self.train_epochs / save_freq))
        numb_steps = num_checkpoints * save_freq

        # Check if the training steps are too small, which happens in the early stage of
        # active learning, and increase it to the default `training_batches`.
        # We observed the model accuracy increases nonlinearly with the dataset size,
        # which means we need a 'minimum' training steps even for an extremely small dataset
        # may have few tens of structures.
        train_config["training"]["numb_steps"] = numb_steps
        if self.train_batches is not None and numb_steps < self.train_batches:
            num_chekpoints = int(np.ceil(self.train_epochs / self.print_epochs))
            new_save_freq = int(np.ceil(self.train_batches / num_chekpoints / min_freq_unit) * min_freq_unit)
            new_numb_steps = new_save_freq * num_chekpoints
            train_config["training"]["save_freq"] = new_save_freq
            train_config["training"]["disp_freq"] = new_save_freq
            train_config["training"]["numb_steps"] = new_numb_steps

        # Write training parameters to deepmd input json
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
                msg = 'Trainer "{}" failed with command "{}" failed in ' "{} with error code {}".format(
                    self.name, command, path, errorcode
                )
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

    def as_dict(self) -> dict:
        """"""
        trainer_params = super().as_dict()
        trainer_params["train_options"] = self.train_options

        return trainer_params


if __name__ == "__main__":
    ...
