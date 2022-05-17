#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import abc
import copy
import pathlib

from typing import Optional
from collections.abc import Iterable


class AbstractDynamics(abc.ABC):

    delete = []
    keyword: Optional[str] = None
    special_keywords = {}

    def __init__(self, calc, directory, *args, **kwargs):

        self.calc = calc
        self.calc.reset()

        self._directory_path = pathlib.Path(directory)

        return

    def set_output_path(self, directory):
        """"""
        # main dynamics dir
        self._directory_path = pathlib.Path(directory)
        # TODO: for repeat, r0, r1, r2
        #self.calc.directory = self._directory_path / self.calc.name
        self.calc.directory = self._directory_path

        # extra files
        #self._logfile_path = self._directory_path / self.logfile
        #self._trajfile_path = self._directory_path / self.trajfile

        return
    
    def reset(self):
        """ remove results stored in dynamics calculator
        """
        self.calc.reset()

        return

    def delete_keywords(self, kwargs):
        """removes list of keywords (delete) from kwargs"""
        for d in self.delete:
            kwargs.pop(d, None)

        return

    def set_keywords(self, kwargs):
        # TODO: rewrite this method
        args = kwargs.pop(self.keyword, [])
        if isinstance(args, str):
            args = [args]
        elif isinstance(args, Iterable):
            args = list(args)

        for key, template in self.special_keywords.items():
            if key in kwargs:
                val = kwargs.pop(key)
                args.append(template.format(val))

        kwargs[self.keyword] = args

        return

    @abc.abstractmethod
    def run(self, atoms, **kwargs):
        """"""


        return 


if __name__ == "__main__":
    pass