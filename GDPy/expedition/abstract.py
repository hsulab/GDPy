#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod
from typing import Union, Callable
from pathlib import Path

class AbstractExplorer(ABC):

    # general settings for IO
    general_parameters = dict(
        ignore_exists = False
    )

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
    
    def __check_dataset(self):

        return
    
    def parse_specorder(self, composition: dict):
        """ determine species order from current structure
            since many systems with different compositions 
            may be explored at the same time
        """
        type_list = []
        for sym, num in composition.items():
            print(sym, num)
            if num != 0:
                type_list.append(sym)

        return type_list
    
    @abstractmethod   
    def icreate(self, exp_name, wd):

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