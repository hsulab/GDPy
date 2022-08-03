#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod
from typing import Union, Callable, Counter, Union, List
from pathlib import Path

import uuid
import time
import json
import warnings

from joblib import Parallel, delayed

from ase import Atoms
from ase.io import read, write

from GDPy import config
from GDPy.machine.machine import SlurmMachine
from GDPy.utils.command import CustomTimer

from GDPy.potential.manager import create_pot_manager


""" abstract class for expedition methods
    - offline exploration
    each one has following procedures:
        creation
            system-name-based-dir
        collection+selection
            sorted-dir
        calculation (includes harvest)
            fp-dir
    a calculator/worker needs defined to drive the exploration
    - on-the-fly exploration
        accelerated series
"""


class AbstractExplorer(ABC):

    name = "expedition"

    # general parameters
    general_params = dict(
        ignore_exists = False
    )

    creation_params = dict(
        calc_dir_name = "tmp_folder"
    )

    collection_params = dict(
        resdir_name = "sorted",
        selection_tags = ["final"]
    )

    calculation_params = dict(

    )

    # system-specific info
    type_map = {}
    type_list = []

    def __init__(self, pm, main_dict):
        """"""
        self.pot_manager = pm
        self._register_type_map(main_dict) # obtain type_list or type_map

        self.explorations = main_dict["explorations"]
        self.init_systems = main_dict["systems"]

        self._parse_general_params(main_dict)

        # for job prefix
        self.job_prefix = ""

        self.njobs = config.NJOBS

        # - parse params
        # --- create
        # --- collect/select
        # --- label/acquire
        pot_dict = main_dict.get("calculation", None)
        if pot_dict is not None:
            self.ref_manager = create_pot_manager(pot_dict)
        else:
            self.ref_manager = None

        return
    
    def _register_type_map(self, input_dict: dict):
        """ create a type map to identify different elements
            should be the same in attached calculator
            the explored system should have some of (not all) these elements
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
                raise RuntimeError("Cant find neither type_map nor type_list.")

        return
    
    def _parse_general_params(self, input_dict: dict):
        """"""
        # parse a few general parameters
        general_params = input_dict.get("general", self.general_params)
        self.ignore_exists = general_params.get("ignore_exists", self.general_params["ignore_exists"])
        print("IGNORE_EXISTS ", self.ignore_exists)

        # database path
        main_database = input_dict.get("dataset", None)
        if main_database is None:
            raise RuntimeError("dataset should not be None...")
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
    
    def run(
        self, 
        operator: Callable[[str, Union[str, Path]], None], 
        working_directory: Union[str, Path]
    ): 
        """create for all explorations"""
        # TODO: automatically check if the current step is finished
        working_directory = Path(working_directory)
        self.job_prefix = working_directory.resolve().name # use resolve to get abspath
        print("job prefix: ", self.job_prefix)
        for exp_name in self.explorations.keys():
            exp_directory = working_directory / exp_name
            # note: check dir existence in sub function
            operator(exp_name, working_directory)

        return

    @abstractmethod   
    def icreate(self, exp_name, wd):

        return

    def icalc(self, exp_name, working_directory, skipped_systems=[]):
        """calculate configurations with reference method"""
        exp_dict = self.explorations[exp_name]

        # - create a calculation machine (vasp, ...)
        if self.ref_manager is None:
            raise RuntimeError("Ref Manager does not exist...")
        else:
            #calc_machine = self.ref_manager.create_machine(
            #    calc_dict
            #)
            calc_machine = self.ref_manager.create_machine()

        # some parameters

        # - create fp main dir
        prefix = working_directory / (exp_name + "-fp")
        if prefix.exists():
            warnings.warn("fp directory exists...", UserWarning)
        else:
            prefix.mkdir(parents=True)

        # - run over systems
        included_systems = exp_dict.get("systems", None)
        if included_systems is not None:
            # - loop over systems
            # TODO: asyncio
            for slabel in included_systems:
                print("--- ")
                sys_frames = [] # NOTE: all frames
                # TODO: make this into system
                if slabel in skipped_systems:
                    continue
                # - result path
                name_path = working_directory / exp_name / slabel

                # - read collected/selected frames
                sorted_path = name_path / self.collection_params["resdir_name"]
                if sorted_path.exists():
                    for tag_name in self.collection_params["selection_tags"]:
                        # - find all selected files
                        # or find final selected that is produced by a composed selector
                        # TODO: if no selected were applied?
                        xyzfiles = list(sorted_path.glob(f"{tag_name}*-selection*.xyz"))
                        if len(xyzfiles) == 1:
                            final_selected_path = xyzfiles[0]
                        else:
                            # assert files have selection order
                            xyzfiles = sorted(xyzfiles, key=lambda x:int(x.name.split(".")[0].split("-")[-1]))
                            final_selected_path = xyzfiles[-1]
                        print(f"found selected structure file {str(final_selected_path)}")
                        # - create input files
                        fp_path = prefix / slabel / tag_name
                        self._prepare_calc_dir(
                            calc_machine,
                            slabel, fp_path, 
                            final_selected_path
                        )
                else:
                    print(f"No candidates to calculate in {str(name_path)}")

        return
    
    def _prepare_calc_dir(
        self, calc_machine, slabel, sorted_fp_path, final_selected_path
    ):
        """ prepare calculation dir
            currently, only vasp is supported
        """
        # - create input file TODO: move this to vasp part
        if not sorted_fp_path.exists():
            sorted_fp_path.mkdir(parents=True)
            # TODO: create
            # - update system-wise parameters
        else:
            # TODO: move harvest function here?
            print(f"{sorted_fp_path} already exists.")

            # - check target calculation structures
            frames_in = read(final_selected_path, ":")
            nframes_in = len(frames_in)

            # - store in database
            # TODO: provide an unified interfac to all type of databases 
            # TODO: offer default xyzfile prefix
            composition = self.init_systems[slabel]["composition"]
            sorted_composition = sorted(composition.items(), key=lambda x:x[0])
            comp_name = "".join([ k+str(v) for k,v in sorted_composition])
            database_path = self.main_database / (self.init_systems[slabel]["prefix"]+"-"+comp_name)

            if not database_path.exists():
                database_path.mkdir(parents=True)

            xyzfile_name = Path("-".join([slabel, self.name, sorted_fp_path.name, self.pot_manager.version]) + ".xyz")

            if xyzfile_name.exists():
                frames_exists = read(xyzfile_name, ":")
                nframes_exists = len(frames_exists)
                if nframes_exists == nframes_in:
                    print("  calculated structures are stored...")
                    return

            # - read results
            with CustomTimer(name="harvest"):
                frames = calc_machine._read_results(sorted_fp_path)
                nframes = len(frames)
                print("nframes calculated: ", nframes)
            
            # - tag every data
            for atoms in frames:
                atoms.info["uuid"] = str(uuid.uuid1())

            if nframes != nframes_in:
                warnings.warn("calculation may not finish...", RuntimeWarning)

            write(database_path / xyzfile_name, frames)

        return

if __name__ == "__main__":
    pass