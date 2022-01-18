#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from abc import ABC
from typing import Union, Callable
from pathlib import Path

class AbstractExplorer(ABC):

    def __init__(self, main_dict: str):
        """"""
        # self.main_dict = main_dict
        self.systems = main_dict["systems"]
        self.explorations = main_dict["explorations"]
        
        return
    
    def register_calculator(self, pm):
        """ use potentila manager
        """

        return

    def run(
        self, 
        operator: Callable[[str, Union[str, Path]], None], 
        working_directory: Union[str, Path]
    ): 
        """create for all explorations"""
        working_directory = Path(working_directory)
        self.job_prefix = working_directory.resolve().name # use resolve to get abspath
        print("job prefix: ", self.job_prefix)
        for exp_name in self.explorations.keys():
            exp_directory = working_directory / exp_name
            # note: check dir existence in sub function
            operator(exp_name, working_directory)

        return

if __name__ == "__main__":
    pass