#!/usr/bin/env python3
# -*- coding: utf-8 -*

import abc
import copy
import os
import pathlib
import subprocess
from typing import Callable

import numpy as np

class TrainingFailed(RuntimeError):

    """Training unexpectedly fails."""

    ...

class AbstractTrainer(abc.ABC):

    #: Name of this trainer.
    name:str = "trainer"

    #: Command to train.
    command:str = None
    
    #: Prefix of input and output.
    prefix:str = "config"

    #: Default output function.
    _print: Callable = print

    def __init__(
        self, config: dict, train_ratio: float=0.9, train_epochs: int=200,
        directory=".", command="train", random_seed: int=None, 
        *args, **kwargs
    ) -> None:
        """"""
        self.command = command
        self.directory = pathlib.Path(directory)
        self.config = config # train model parameters

        self.train_ratio = train_ratio
        self.train_epochs = train_epochs

        if random_seed is None:
            random_seed = np.random.randint(0, 10000)
        self.random_seed = random_seed
        self.rng = np.random.default_rng(seed=random_seed)

        return
    
    def train(self, dataset):
        """"""
        if self.command is None:
            raise TrainingFailed(
                "Please set ${} environment variable "
                .format("GDP_" + self.name.upper() + "_COMMAND") +
                "or supply the command keyword")
        command = self.command
        if "PREFIX" in command:
            command = command.replace("PREFIX", self.prefix)
        
        # TODO: ...
        if not self.directory.exists():
            self.directory.mkdir(parents=True, exist_ok=True)
        self.write_input(dataset, batchsizes=4, reduce_system=False)

        try:
            proc = subprocess.Popen(command, shell=True, cwd=self.directory)
        except OSError as err:
            msg = "Failed to execute `{}`".format(command)
            raise TrainingFailed(msg) from err

        errorcode = proc.wait()

        if errorcode:
            path = os.path.abspath(self.directory)
            msg = ('Calculator "{}" failed with command "{}" failed in '
                   '{} with error code {}'.format(self.name, command,
                                                  path, errorcode))
            raise TrainingFailed(msg)

        return
    
    @abc.abstractmethod
    def write_input(self, dataset, batchsizes, reduce_system: bool=False):
        """Write inputs for training.

        Args:
            reduce_system: Whether merge structures.

        """

        return
    
    @abc.abstractmethod
    def read_convergence(self):
        """"""

        return
    
    def as_dict(self) -> dict:
        """"""
        trainer_params = {}
        trainer_params["name"] = self.name
        trainer_params["config"] = self.config
        trainer_params["command"] = self.command
        trainer_params["train_ratio"] = self.train_ratio
        trainer_params["train_epochs"] = self.train_epochs
        trainer_params["random_seed"] = self.random_seed

        trainer_params = copy.deepcopy(trainer_params)

        return trainer_params


if __name__ == "__main__":
    ...