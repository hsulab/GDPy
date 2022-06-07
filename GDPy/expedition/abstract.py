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

    type_map = {}
    type_list = []

    def __init__(self, main_dict: str):
        """"""
        # self.main_dict = main_dict
        self.systems = main_dict["systems"]
        self.explorations = main_dict["explorations"]
        
        return
    
    def _register_type_map(self, input_dict: dict):
        """ create a type map to identify different elements
            should be the same in attached calculator
        """
        type_map_ = input_dict.get("type_map", None)
        if type_map_ is not None:
            self.type_map = type_map_
            type_list_ = [item[0] for item in sorted(type_map_.items(), key=lambda x: x[1])]
            self.type_list = type_list_
        else:
            type_list_ = input_dict.get("type_list", None)
            if type_list_ is not None:
                self.type_list = type_list_
                type_map_ = {}
                for i, s in enumerate(type_list_):
                    type_map_[s] = i
                self.type_map = type_map_
            else:
                raise RuntimeError("Cant find neither type_map or type_list.")

        return
    
    def _parse_general_params(self, input_dict: dict):
        """"""
        # parse a few general parameters
        general_params = input_dict.get("general", self.general_params)
        self.ignore_exists = general_params.get("ignore_exists", self.general_params["ignore_exists"])
        print("IGNORE_EXISTS ", self.ignore_exists)

        # database path
        main_database = input_dict.get("dataset", None) #"/users/40247882/scratch2/PtOx-dataset"
        if main_database is None:
            raise ValueError("dataset should not be None")
        else:
            self.main_database = Path(main_database)

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