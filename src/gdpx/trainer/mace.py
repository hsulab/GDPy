#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import copy
import itertools
import pathlib
from typing import Optional

from ase.io import write

from gdpx.dataloader.mace import MaceDataloader

from .trainer import BasePotentialTrainer


class MaceTrainer(BasePotentialTrainer):

    name = "mace"
    command = "mace_run_train"
    freeze_command = ""

    _train_fname = "_train.xyz"
    _test_fname = "_test.xyz"

    #: Flag indicates that the training is finished properly.
    CONVERGENCE_FLAG: str = "Done"

    def __init__(
        self,
        config: dict,
        type_list: Optional[list[str]] = None,
        train_epochs: int = 200,
        print_epochs: int = 5,
        directory=".",
        command="python ./run_train.py",
        freeze_command="python ./run_train.py",
        random_seed: int = None,
        *args,
        **kwargs,
    ) -> None:
        """"""
        super().__init__(
            config,
            type_list,
            train_epochs,
            print_epochs,
            directory,
            command,
            freeze_command,
            random_seed,
            *args,
            **kwargs,
        )

        self._type_list = type_list

        return

    def _resolve_train_command(self, init_model=None, *args, **kwargs) -> str:
        """"""

        return self.command

    def freeze(self):
        """"""
        # models = list((self.directory/"checkpoints").glob("*.model"))
        use_swa = self.config.get("swa", False)
        if not use_swa:
            model_fpath = self.directory / ("{}.model".format(self.config["name"]))
        else:
            model_fpath = self.directory / ("{}_swa.model".format(self.config["name"]))

        return model_fpath

    def _resolve_freeze_command(self, *args, **kwargs) -> str:
        """"""

        return self.freeze_command

    @property
    def frozen_name(self):
        """"""

        return f"{self.name}.model"

    def _update_config(self, dataset, *args, **kwargs) -> dict:
        """"""
        # - update config
        train_config = copy.deepcopy(self.config)
        train_config["name"] = train_config.get("name", "mace")
        train_config["seed"] = self.random_seed
        train_config["train_file"] = str(dataset.train_file)
        train_config["valid_file"] = str(dataset.test_file)
        train_config["valid_fraction"] = 0.0
        test_file = train_config.get("test_file")
        if test_file is not None:
            train_config["test_file"] = str(pathlib.Path(test_file).resolve())
        else:
            train_config["test_file"] = str(dataset.test_file)

        # TODO: plus one to save the final checkpoint?
        train_config["max_num_epochs"] = self.train_epochs
        train_config["eval_interval"] = self.print_epochs
        train_config["batch_size"] = dataset.batchsize

        swa = train_config.get("swa", False)
        if swa:
            start_swa = train_config.get("start_swa", -1)
            if not (0 < start_swa < self.train_epochs):
                raise RuntimeError(f"{start_swa = } must be smaller than {self.train_epochs = }")
        else:
            ...

        # - misc
        import torch

        train_config["device"] = "cuda" if torch.cuda.is_available() else "cpu"

        restart_latest = train_config.get("restart_latest", True)
        train_config["restart_latest"] = restart_latest

        init_latest = train_config.get("init_latest", None)
        if init_latest is not None and init_latest:
            train_config["init_latest"] = init_latest
            del train_config["restart_latest"]

        train_config["save_cpu"] = True

        # - pop unused...
        train_config.pop("log_dir", None)
        train_config.pop("model_dir", None)
        train_config.pop("checkpoints_dir", None)
        train_config.pop("results_dir", None)

        return train_config

    def get_checkpoint(self):
        """"""

        return pathlib.Path(self.directory / "checkpoints").resolve()

    def _train_from_the_restart(self, dataset, init_model) -> str:
        """Train from the restart.

        Args:
            init_model: The path of the checkpoints folder.

        """
        # if init_model is not None:
        #     raise NotImplementedError(
        #         f"{self.name} does not support initialising from a previous model."
        #     )

        def _add_command_options(command, config) -> str:
            """"""
            # - convert to command line options...
            command = command + " "
            for k, v in config.items():
                if isinstance(v, bool):
                    if v:
                        command += f"--{k}  "
                elif isinstance(v, int) or isinstance(v, float):
                    command += f"--{k}={str(v)}  "
                else:
                    command += f"--{k}='{str(v)}'  "

            return command

        def _check_latest_checkpoint(ckpt_dir, model_name) -> Optional[int]:
            """"""
            ckpts = [p for p in ckpt_dir.glob(f"{model_name}*")]
            # ckpt_models = [c for c in ckpts if c.name.endswith(".model")]
            ckpt_models = [c for c in ckpts if c.name.endswith(".pt")]
            num_ckpts = len(ckpt_models)
            assert num_ckpts <= 1
            if num_ckpts == 1:
                ckpt_model = ckpt_models[0]
                prev_seed = int(ckpt_model.name.split("-")[1].split("_")[0])
            else:
                prev_seed = None

            return prev_seed

        def _get_latest_checkpoint(ckpt_dir, model_name):
            """"""
            ckpts = [p for p in ckpt_dir.glob(f"{model_name}*")]
            ckpt_models = [c for c in ckpts if c.name.endswith(".pt")]
            num_ckpts = len(ckpt_models)
            assert num_ckpts <= 1
            if num_ckpts == 1:
                ckpt_model = ckpt_models[0]
            else:
                ckpt_model = None

            return ckpt_model

        # Check dataset type and convert it if necessary
        dataset = self._prepare_dataset(dataset)

        train_config = self._update_config(dataset)

        # make command
        raw_command = self._train_from_the_scratch(dataset, init_model)

        ckpt_dir = self.directory / "checkpoints"
        if not self.directory.exists():
            if init_model is not None:
                model_name = train_config["name"]
                init_model = pathlib.Path(init_model)
                assert init_model.name == "checkpoints"
                self._print(f"init_model: {str(init_model)}")
                ckpt_path = _get_latest_checkpoint(init_model, model_name)
                if ckpt_path is not None:
                    ckpt_dir.mkdir()
                    curr_seed = train_config["seed"]
                    (ckpt_dir / f"{model_name}_run-{curr_seed}_epoch-0.pt").symlink_to(ckpt_path)
                    train_config.pop("restart_latest", None)
                    train_config["init_latest"] = True
                else:
                    self._print(f"FAILED to init from `{str(init_model)}`.")
            else:
                # train from the scratch and no config needs update
                ...
        else:
            if ckpt_dir.exists():
                # continue from the latest checkpoint
                prev_seed = _check_latest_checkpoint(ckpt_dir, train_config["name"])
                self._print(f"{prev_seed =}")
                if prev_seed is not None:
                    train_config["seed"] = prev_seed
                    train_config.pop("init_latest", None)
                    train_config["restart_latest"] = True
            else:
                if init_model is not None:
                    model_name = train_config["name"]
                    init_model = pathlib.Path(init_model)
                    assert init_model.name == "checkpoints"
                    self._print(f"init_model: {str(init_model)}")
                    ckpt_path = _get_latest_checkpoint(init_model, model_name)
                    if ckpt_path is not None:
                        ckpt_dir.mkdir()
                        curr_seed = train_config["seed"]
                        (ckpt_dir / f"{model_name}_run-{curr_seed}_epoch-0.pt").symlink_to(ckpt_path)
                        train_config.pop("restart_latest", None)
                        train_config["init_latest"] = True
                    else:
                        self._print(f"FAILED to init from `{str(init_model)}`.")
                else:
                    # train from the scratch and no config needs update
                    ...

        command = _add_command_options(raw_command, train_config)

        return command

    def _prepare_dataset(self, dataset, *args, **kwargs):
        """Prepare a reann dataset for training.

        Currently, it only supports converting xyz dataset.

        """
        self._print(f"{dataset = }")
        # NOTE: make sure the dataset path exists, sometimes it will be access
        #       before training to create a shared dataset
        self.directory.mkdir(parents=True, exist_ok=True)

        if not isinstance(dataset, MaceDataloader):
            set_names, train_frames, test_frames, adjusted_batchsizes = dataset.split_train_and_test()

            # NOTE: reann does not support split-system training,
            #       so we need merge all structures into one list
            train_frames = itertools.chain(*train_frames)
            write(self.directory / self._train_fname, train_frames)

            test_frames = itertools.chain(*test_frames)
            write(self.directory / self._test_fname, test_frames)

            dataset = MaceDataloader(
                train_file=self.directory / self._train_fname,
                test_file=self.directory / self._test_fname,
                directory=self.directory,
                batchsize=dataset.batchsize,
            )
        else:
            ...

        return dataset

    def write_input(self, dataset, *args, **kwargs):
        """Convert dataset to the target format and write the configuration file if it has."""
        self._print(f"write {self.name} inputs...")

        # - convert dataset
        dataset = self._prepare_dataset(dataset)

        # - input config
        #   mace uses command line options

        return

    def read_convergence(self):
        """"""
        super().read_convergence()

        converged = False
        logs = list((self.directory / "logs").glob("*.log"))
        assert len(logs) == 1, "There should be only one log."
        with open(logs[0], "r") as fopen:
            lines = fopen.readlines()
        if self.CONVERGENCE_FLAG in lines[-1]:
            converged = True

        return converged


if __name__ == "__main__":
    ...
