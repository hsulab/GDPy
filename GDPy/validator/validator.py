#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import abc
import pathlib
import logging

from typing import NoReturn, Union


class AbstractValidator(abc.ABC):

    _directory = pathlib.Path.cwd()

    restart = True

    def __init__(self, directory: Union[str,pathlib.Path], task_params: dict, pot_worker=None):
        """
        """
        self.directory = pathlib.Path(directory)

        self.task_params = task_params

        self.pm = pot_worker.potter
        self.calc = self.pm.calc

        self._init_logger()

        return
    
    @property
    def directory(self):
        """"""

        return self._directory

    @directory.setter
    def directory(self, directory_):
        """"""
        # - create main dir
        directory_ = pathlib.Path(directory_)
        if not directory_.exists():
            directory_.mkdir() # NOTE: ./tmp_folder
        else:
            pass
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
    
    def __parse_outputs(self, input_dict: dict) -> NoReturn:
        """ parse and create ouput folders and files
        """
        self.output_path = pathlib.Path(input_dict.get("output", "miaow"))
        if not self.output_path.exists():
            self.output_path.mkdir(parents=True)

        return

    @abc.abstractmethod
    def run(self, *args, **kwargs):
        return


if __name__ == "__main__":
    pass