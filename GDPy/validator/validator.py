#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import abc
import pathlib

from typing import NoReturn, Union


class AbstractValidator(abc.ABC):

    _directory = pathlib.Path.cwd()

    def __init__(self, directory: Union[str,pathlib.Path], task_params: dict, pot_worker=None):
        """
        """
        self.directory = pathlib.Path(directory)

        self.task_params = task_params

        self.pm = pot_worker.potter
        self.calc = self.pm.calc

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