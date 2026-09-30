#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import copy
import json
import os
import pathlib
import subprocess
from typing import Optional

import numpy as np

from gdpx.data.loaders.deepmd import DeepmdDataloader

from ...training import BasePotentialTrainer, FreezingFailed
from .convert import convert_groups


class DeepmdTrainer(BasePotentialTrainer):

    name = "deepmd"
    requires_dataloader = True
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

    @property
    def model_family(self) -> Optional[str]:
        """Return the DeepMD model family when it selects a special CLI path."""
        model = self.config.get("model", {})
        model_type = str(model.get("type", "")).lower()
        descriptor = model.get("descriptor", {})
        descriptor_type = (
            str(descriptor.get("type", "")).lower()
            if isinstance(descriptor, dict)
            else ""
        )
        preset = str(model.get("preset", "")).lower()

        if model_type == "dpa4c" or descriptor_type == "dpa4c" or preset.startswith("dpa4c-"):
            return "dpa4c"
        if model_type in {"dpa4", "sezm"} or descriptor_type in {"dpa4", "sezm"}:
            return "dpa4"
        return None

    def _resolve_dp_command(self, command: str) -> str:
        """Add the DeepMD backend selector required by modern DPA4 models."""
        backend_flag = {"dpa4": "--pt", "dpa4c": "--pt-expt"}.get(self.model_family)
        if backend_flag is not None and backend_flag not in command.split():
            command = f"{command} {backend_flag}"
        return command

    @property
    def checkpoint_name(self) -> str:
        """Return the checkpoint used to restart or export the configured model."""
        save_ckpt = self.config.get("training", {}).get("save_ckpt", "model.ckpt")
        if self.model_family in {"dpa4", "dpa4c"} and not save_ckpt.endswith(".pt"):
            save_ckpt += ".pt"
        return save_ckpt

    def _resolve_train_command(self, init_model=None):
        """"""
        train_command = self._resolve_dp_command(self.command)

        # - add options
        command = "{} train {}.json {} ".format(train_command, self.name, self.train_options)
        if init_model is not None:
            init_model_path = pathlib.Path(init_model).resolve()
            if init_model_path.name.endswith(".pb"):
                command += " --init-frz-model {}".format(str(init_model_path))
            elif init_model_path.name.endswith("model.ckpt"):
                command += " --init-model {}".format(str(init_model_path))
            elif self.model_family in {"dpa4", "dpa4c"} and init_model_path.suffix == ".pt":
                command += " --finetune {}".format(str(init_model_path))
            else:
                raise RuntimeError(f"Unknown init_model {str(init_model_path)}.")
        command += " 2>&1 > {}.out".format(self.name)

        return command

    def _resolve_freeze_command(self, *args, **kwargs):
        """"""
        freeze_command = self._resolve_dp_command(self.freeze_command)

        # - add options
        if self.model_family in {"dpa4", "dpa4c"}:
            export_options = " --lower-kind graph" if self.model_family == "dpa4c" else ""
            command = "{} freeze -c {} -o {}{} 2>&1 >> {}.out".format(
                freeze_command,
                self.checkpoint_name,
                pathlib.Path(self.frozen_name).stem,
                export_options,
                self.name,
            )
        else:
            command = "{} freeze -o {} 2>&1 >> {}.out".format(freeze_command, self.frozen_name, self.name)

        return command

    def _resolve_compress_command(self, *args, **kwargs):
        """"""
        compress_command = self._resolve_dp_command(self.command)

        # - add options
        command = "{} compress -i {} -o {} 2>&1 >> {}.out".format(
            compress_command, self.frozen_name, self.compressed_name, self.name
        )

        return command

    @property
    def frozen_name(self):
        """"""
        if self.model_family in {"dpa4", "dpa4c"}:
            return f"{self.name}.pt2"
        return f"{self.name}.pb"

    @property
    def compressed_name(self):
        """Return the deployed artifact name after model compression."""
        if self.model_family == "dpa4c":
            return f"{self.name}-c.pt2"
        return f"{self.name}-c.pb"

    def _train_from_the_restart(self, dataset, init_model):
        """Train from the restart"""
        if not self.directory.exists():
            command = self._train_from_the_scratch(dataset, init_model)
        else:
            ckpt_info = self.directory / "checkpoint"
            if ckpt_info.exists() and ckpt_info.stat().st_size != 0:
                # TODO: check if the ckpt model exists?
                command = f"{self._resolve_dp_command(self.command)} train {self.name}.json "
                command += f"--restart {self.checkpoint_name}"
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

        descriptor = train_config["model"].get("descriptor")
        if isinstance(descriptor, dict):
            descriptor["seed"] = self.rng.integers(0, 10000, dtype=int)
        fitting_net = train_config["model"].get("fitting_net")
        if isinstance(fitting_net, dict):
            fitting_net["seed"] = self.rng.integers(0, 10000, dtype=int)

        training = train_config.setdefault("training", {})
        training_data = training.setdefault("training_data", {})
        training_data["systems"] = [x for x in dataset.train_sys_dirs]
        training_data["batch_size"] = dataset.batchsizes

        # verify validation_data
        validation_data, validation_batchsizes = [], []
        for v_system, v_batchsize in zip(dataset.valid_sys_dirs, dataset.batchsizes):
            if v_system != "None":  # None will be saved to a string before
                validation_data.append(v_system)
                validation_batchsizes.append(v_batchsize)
        if validation_data:
            validation_config = training.setdefault("validation_data", {})
            validation_config["systems"] = validation_data
            validation_config["batch_size"] = validation_batchsizes
        else:
            training.pop("validation_data", None)

        training["seed"] = self.rng.integers(0, 10000, dtype=int)

        # GDP owns the training duration.  DeePMD accepts either epoch- or
        # step-based stopping, but rejects a configuration containing both.
        training.pop("num_epochs", None)

        # Determine `numb_steps`
        min_freq_unit = 100.0
        save_freq = int(np.ceil(dataset.cum_batchsizes * self.print_epochs / min_freq_unit) * min_freq_unit)
        training["save_freq"] = save_freq

        # NOTE: Currently, we check whether the training is fininished by steps in lcurve.out.
        #       Thus, we need make sure the last step (numb_steps) is displayed in lcurve.out
        #       by making numb_steps can be divided by disp_freq.
        training["disp_freq"] = save_freq

        numb_steps = dataset.cum_batchsizes * self.train_epochs
        num_checkpoints = int(np.ceil(dataset.cum_batchsizes * self.train_epochs / save_freq))
        numb_steps = num_checkpoints * save_freq

        # Check if the training steps are too small, which happens in the early stage of
        # active learning, and increase it to the default `training_batches`.
        # We observed the model accuracy increases nonlinearly with the dataset size,
        # which means we need a 'minimum' training steps even for an extremely small dataset
        # may have few tens of structures.
        training["numb_steps"] = numb_steps
        if self.train_batches is not None and numb_steps < self.train_batches:
            num_chekpoints = int(np.ceil(self.train_epochs / self.print_epochs))
            new_save_freq = int(np.ceil(self.train_batches / num_chekpoints / min_freq_unit) * min_freq_unit)
            new_numb_steps = new_save_freq * num_chekpoints
            training["save_freq"] = new_save_freq
            training["disp_freq"] = new_save_freq
            training["numb_steps"] = new_numb_steps

        # Write training parameters to deepmd input json
        with open(self.directory / f"{self.name}.json", "w") as fopen:
            json.dump(train_config, fopen, indent=2)

        return

    def freeze(self):
        """"""
        # - freeze model
        frozen_model = super().freeze()

        # DPA4 exports directly to an AOTInductor .pt2 archive and does not
        # support compression.  DPA4C uses the exportable graph route and its
        # compressed .pt2 is the preferred deployment artifact.
        if self.model_family == "dpa4":
            return frozen_model

        # - compress model
        compressed_model = (self.directory / self.compressed_name).absolute()
        if frozen_model.exists() and not compressed_model.exists():
            command = self._resolve_compress_command()
            compression_error = None
            try:
                proc = subprocess.Popen(command, shell=True, cwd=self.directory)
                errorcode = proc.wait()
                if errorcode:
                    compression_error = RuntimeError(f"error code {errorcode}")
            except (OSError, RuntimeError) as err:
                compression_error = err

            if compression_error is not None:
                self._print("Failed to compress model.")
                path = os.path.abspath(self.directory)
                msg = 'Trainer "{}" failed to compress with command "{}" in {}: {}'.format(
                    self.name, command, path, compression_error
                )
                if self.model_family == "dpa4c":
                    raise FreezingFailed(msg) from compression_error
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

        Check DeepMD training progress against its normalized configuration.

        DeepMD writes ``out.json`` after resolving aliases such as
        ``num_epochs`` into a concrete number of steps.  Prefer that file and
        fall back to GDP's generated input.  This method deliberately checks
        training only: a completed DPA4 checkpoint remains recoverable when a
        later AOTInductor export fails and should not be trained again.

        """
        self._print(f"check {self.name} training convergence...")
        converged = False

        numb_steps = None
        for config_name in ("out.json", f"{self.name}.json"):
            config_path = self.directory / config_name
            if not config_path.exists():
                continue
            try:
                with open(config_path, "r") as fopen:
                    input_json = json.load(fopen)
                numb_steps = input_json.get("training", {}).get("numb_steps")
            except (OSError, json.JSONDecodeError, TypeError):
                continue
            if numb_steps is not None:
                break

        lcurve_out = self.directory / "lcurve.out"
        curr_steps = None
        if numb_steps is not None and lcurve_out.exists():
            try:
                with open(lcurve_out, "r") as fopen:
                    for line in reversed(fopen.readlines()):
                        fields = line.strip().split()
                        if not fields or fields[0].startswith("#"):
                            continue
                        try:
                            curr_steps = int(fields[0])
                        except ValueError:
                            continue
                        break
            except OSError:
                curr_steps = None

        if curr_steps is not None:
            converged = curr_steps >= int(numb_steps)
            self._debug(f"{curr_steps} >=? {numb_steps}")
        elif numb_steps is not None:
            self._print("No valid training step was found in `lcurve.out`.")

        return converged

    def as_dict(self) -> dict:
        """"""
        trainer_params = super().as_dict()
        trainer_params["train_options"] = self.train_options

        return trainer_params


if __name__ == "__main__":
    ...
