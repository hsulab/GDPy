#!/usr/bin/env python3
# -*- coding: utf-8 -*

import abc
import copy
import os
import pathlib
import subprocess
from typing import Union, Callable, List

import numpy as np

from .. import config

class TrainingFailed(RuntimeError):

    """Training unexpectedly fails."""

    ...

class FreezingFailed(RuntimeError):

    """Freezing unexpectedly fails."""

    ...

class AbstractTrainer(abc.ABC):

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

    #: Default output function.
    _print: Callable = config._print

    #: Default debug function.
    _debug: Callable = config._debug

    #: Working directory.
    _directory: Union[str,pathlib.Path] = "./"

    def __init__(
        self, config: dict, type_list: List[str]=None, train_epochs: int=200,
        directory=".", command="train", freeze_command="freeze", random_seed: int=None, 
        *args, **kwargs
    ) -> None:
        """"""
        self.command = command
        if freeze_command is None:
            self.freeze_command = self.command
        else:
            self.freeze_command = freeze_command

        self.directory = directory
        self.config = config # train model parameters

        # - TODO: sync type_list

        self.train_epochs = train_epochs

        if random_seed is None:
            random_seed = np.random.randint(0, 10000)
        self.random_seed = random_seed
        self.rng = np.random.default_rng(seed=random_seed)

        return
    
    @property
    def directory(self):
        """Directory should always be absolute."""
        
        return self._directory
    
    @directory.setter
    def directory(self, directory: Union[str,pathlib.Path]):
        """"""
        self._directory = pathlib.Path(directory).resolve()

        return

    @property
    def type_list(self):
        """"""

        return self._type_list
    
    def _update_config(self, dataset, *args, **kwargs):
        """Some configuration parameters can only be determined after checking the dataset.

        For example, MACE... This function will modify parameters in `self.config`.

        """

        return

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
    
    def train(self, dataset, init_model=None, *args, **kwargs):
        """"""
        self._update_config(dataset=dataset)

        command = self._resolve_train_command(init_model)
        if command is None:
            raise TrainingFailed(
                "Please set ${} environment variable "
                .format("GDP_" + self.name.upper() + "_COMMAND") +
                "or supply the command keyword")
        self._print(f"COMMAND: {command}")
        
        # TODO: ...
        # TODO: restart?
        if not self.directory.exists():
            self.directory.mkdir(parents=True, exist_ok=True)
        self.write_input(dataset, reduce_system=False)

        try:
            proc = subprocess.Popen(command, shell=True, cwd=self.directory)
        except OSError as err:
            msg = "Failed to execute `{}`".format(command)
            raise TrainingFailed(msg) from err

        errorcode = proc.wait()

        if errorcode:
            path = os.path.abspath(self.directory)
            msg = ('Trainer "{}" failed with command "{}" failed in '
                   '{} with error code {}'.format(self.name, command,
                                                  path, errorcode))
            raise TrainingFailed(msg)

        return
    
    def freeze(self):
        """Freeze trained model and return the model path."""
        frozen_model = (self.directory/self.frozen_name).resolve()
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
                msg = ('Trainer "{}" failed with command "{}" failed in '
                       '{} with error code {}'.format(self.name, command,
                                                      path, errorcode))
                raise FreezingFailed(msg)
        else:
            ...

        return self.directory/self.frozen_name
    
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
        trainer_params["random_seed"] = self.random_seed

        trainer_params = copy.deepcopy(trainer_params)

        return trainer_params


if __name__ == "__main__":
    ...