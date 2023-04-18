#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import abc
import pathlib
import logging

from typing import NoReturn, Union

from GDPy.computation.worker.drive import DriverBasedWorker


class AbstractValidator(abc.ABC):

    _directory = pathlib.Path.cwd()

    restart = True

    def __init__(
        self, directory: Union[str,pathlib.Path], task_params: dict
    ):
        """
        """
        self.directory = pathlib.Path(directory)

        self.task_params = task_params

        #self.worker = pot_worker
        #self.pm = pot_worker.potter
        #self.calc = self.pm.calc

        #self._init_logger()

        return
    
    @property
    def directory(self):
        """"""

        return self._directory

    @directory.setter
    def directory(self, directory_):
        """"""
        directory_ = pathlib.Path(directory_)
        self._directory = directory_

        return

    def _init_logger(self):
        """"""
        self.logger = logging.getLogger(__name__)

        log_level = logging.INFO

        self.logger.setLevel(log_level)

        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        # - screen
        if not self.logger.hasHandlers():
            ch = logging.StreamHandler()
            ch.setLevel(log_level)
            #ch.setFormatter(formatter)
            self.logger.addHandler(ch)

        # - file
        working_directory = self.directory
        log_fpath = working_directory / (self.__class__.__name__+".out")

        if self.restart:
            fh = logging.FileHandler(filename=log_fpath, mode="a")
        else:
            fh = logging.FileHandler(filename=log_fpath, mode="w")
        fh.setLevel(log_level)
        #fh.setFormatter(formatter)
        self.logger.addHandler(fh)

        return

    @abc.abstractmethod
    def run(self, *args, **kwargs):
        """"""
        if not self.directory.exists():
            self.directory.mkdir(parents=True)

        #if self.logger is not None:
        #    self._pfunc = self.logger.info
        #self.pfunc(f"@@@{self.__class__.__name__}")
        self._init_logger()

        return


if __name__ == "__main__":
    ...