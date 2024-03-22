#!/usr/bin/env python3
# -*- coding: utf-8 -*


import abc
import copy
import os
import pathlib
import subprocess

from typing import Union, Callable, List

from ..core.node import AbstractNode


class TrainingFailed(RuntimeError):
    """Training unexpectedly fails."""

    ...


class FreezingFailed(RuntimeError):
    """Freezing unexpectedly fails."""

    ...


class AbstractTrainer(AbstractNode):

    #: Name of this trainer.
    name: str = "trainer"

    #: The path of the command executable.
    command: str = None

    #: Command to freeze/deploy.
    freeze_command: str = None

    #: Type list e.g. [C, H, O].
    _type_list: List[str] = None

    #: Prefix of input file.
    prefix: str = "config"

    def __init__(
        self,
        config: dict,
        type_list: List[str] = None,
        train_epochs: int = 200,
        print_epochs: int = 5,
        directory=".",
        command="train",
        freeze_command="freeze",
        random_seed: Union[int, dict] = None,
        *args,
        **kwargs,
    ) -> None:
        """"""
        super().__init__(directory=directory, random_seed=random_seed)
        self.command = command
        if freeze_command is None:
            self.freeze_command = self.command
        else:
            self.freeze_command = freeze_command

        self.directory = directory
        self.config = config  # train model parameters

        # - TODO: sync type_list

        self.train_epochs = train_epochs
        self.print_epochs = print_epochs

        return

    @property
    def directory(self):
        """Directory should always be absolute."""

        return self._directory

    @directory.setter
    def directory(self, directory: Union[str, pathlib.Path]):
        """"""
        self._directory = pathlib.Path(directory).resolve()

        return

    @property
    def type_list(self):
        """"""

        return self._type_list

    @abc.abstractmethod
    def _resolve_train_command(self, *args, **kwargs):
        """"""

        return

    @abc.abstractmethod
    def _resolve_freeze_command(self, *args, **kwargs):
        """"""

        return

    @property
    @abc.abstractmethod
    def frozen_name(self):
        """"""
        ...

    def _train_from_the_scratch(self, dataset, init_model):
        """Train from the scratch."""
        command = self._resolve_train_command(init_model)
        if command is None:
            raise TrainingFailed(
                f"Please supply the command keyword for {self.name.upper()}."
            )

        if not self.directory.exists():
            self.directory.mkdir(parents=True, exist_ok=True)
        self.write_input(dataset)

        return command

    def train(self, dataset, init_model=None, *args, **kwargs):
        """"""
        if not hasattr(self, "_train_from_the_restart"):
            command = self._train_from_the_scratch(dataset, init_model)
        else:
            command = self._train_from_the_restart(dataset, init_model)
        self._print(f"TRAINING COMMAND: {command}")

        try:
            proc = subprocess.Popen(command, shell=True, cwd=self.directory)
        except OSError as err:
            msg = "Failed to execute `{}`".format(command)
            raise TrainingFailed(msg) from err

        errorcode = proc.wait()

        if errorcode:
            path = os.path.abspath(self.directory)
            msg = (
                'Trainer "{}" failed with command "{}" failed in '
                "{} with error code {}".format(self.name, command, path, errorcode)
            )
            raise TrainingFailed(msg)

        return

    def freeze(self):
        """Freeze trained model and return the model path."""
        frozen_model = (self.directory / self.frozen_name).resolve()
        if not frozen_model.exists():
            command = self._resolve_freeze_command()
            try:
                proc = subprocess.Popen(command, shell=True, cwd=self.directory)
            except OSError as err:
                msg = "Failed to execute `{}`".format(command)
                raise FreezingFailed(msg) from err

            errorcode = proc.wait()

            if errorcode:
                path = os.path.abspath(self.directory)
                msg = (
                    'Trainer "{}" failed with command "{}" failed in '
                    "{} with error code {}".format(self.name, command, path, errorcode)
                )
                raise FreezingFailed(msg)
        else:
            ...

        return self.directory / self.frozen_name

    @abc.abstractmethod
    def write_input(self, dataset, *args, **kwargs):
        """Convert dataset to the target format and write the configuration file if it has."""

        return

    @abc.abstractmethod
    def read_convergence(self) -> bool:
        """"""

        return

    def as_dict(self) -> dict:
        """"""
        trainer_params = {}
        trainer_params["name"] = self.name
        trainer_params["type_list"] = self.type_list
        trainer_params["config"] = self.config
        trainer_params["command"] = self.command
        trainer_params["freeze_command"] = self.freeze_command
        trainer_params["train_epochs"] = self.train_epochs

        # NOTE: self.random_seed may be changed thus we save the init one
        trainer_params["random_seed"] = self.init_random_seed

        trainer_params = copy.deepcopy(trainer_params)

        return trainer_params


if __name__ == "__main__":
    ...
