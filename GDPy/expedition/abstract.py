#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod
from typing import Union, Callable, Counter, Union, List, NoReturn
from pathlib import Path

import os
import copy
import shutil
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

from GDPy.potential.manager import create_potter
from GDPy.selector.abstract import create_selector


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

    # name of expedition
    name = "expedition"

    parameters = dict()

    # general parameters
    general_params = dict(
        ignore_exists = False
    )

    creation_params = dict(
        calc_dir_name = "tmp_folder"
    )

    collection_params = dict(
        resdir_name = "sorted",
        traj_period = 1,
        selection_tags = ["final"]
    )

    calculation_params = dict(

    )

    label_params = dict(
        check_parity = True
    )

    # system-specific info
    type_map = {}
    type_list = []

    def __init__(self, potter, main_dict):
        """"""
        self.pot_manager = potter
        self._register_type_map(main_dict) # obtain type_list or type_map

        self.explorations = main_dict["explorations"]
        self.init_systems = main_dict["systems"]

        self.machine_dict = main_dict.get("machines", {})

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
            self.ref_manager = create_potter(pot_dict)
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
    
    def register_machine(self, pm):
        """ create machine
        """

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

    def icreate(self, exp_name: str, working_directory: Union[str,Path]) -> NoReturn:
        """ create expedition tasks and gather results
        """
        # - a few info
        exp_dict = self.explorations[exp_name]
        included_systems = exp_dict.get("systems", None)

        actions, selector = self._prior_create(exp_dict)

        if included_systems is not None:
            for slabel in included_systems:
                print(f"----- Explore System {slabel} -----")

                # - prepare output directory
                res_dpath = working_directory / exp_name / slabel
                if not res_dpath.exists():
                    res_dpath.mkdir(parents=True)
                #else:
                #    print(f"  {res_dpath.name} exists, so next...")
                #    continue

                # - read substrate
                self.step_dpath = self._make_step_dir(res_dpath, "init")
                frames, cons_text = self._read_structure(slabel)

                # --- update cons text
                if isinstance(actions["driver"], list):
                    for driver in actions["driver"]:
                        driver.run_params.update(constraint=cons_text)
                else:
                    actions["driver"].run_params.update(constraint=cons_text)

                # - run exploration
                # NOTE: check status?
                # TODO: make this a while loop
                status = ""
                while status != "finished":
                    status = self._check_create_status(res_dpath)
                    print(f"===== current status: {status}  =====")
                    self.step_dpath = self._make_step_dir(res_dpath, status)
                    if status == "create":
                        with open(res_dpath/"CREATE_RUNNING", "a") as fopen:
                            fopen.write(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
                        self._single_create(res_dpath, frames, actions)
                        with open(res_dpath/"CREATE_FINISHED", "a") as fopen:
                            fopen.write(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
                        os.remove(res_dpath/"CREATE_RUNNING")
                    elif status == "collect":
                        with open(res_dpath/"COLLECT_RUNNING", "a") as fopen:
                            fopen.write(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
                        self._single_collect(res_dpath, frames, actions, selector)
                        with open(res_dpath/"COLLECT_FINISHED", "a") as fopen:
                            fopen.write(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
                        os.remove(res_dpath/"COLLECT_RUNNING")
                    elif status == "finished":
                        pass
                    else:
                        raise RuntimeError("Unknown status for exploration...")
        else:
            # NOTE: no systems to explore
            pass

        self._post_create()

        return
    
    def _make_step_dir(self, res_dpath: Path, status: str) -> Path:
        """"""
        if status not in ["init", "create", "collect", "select"]:
            return

        # - create collect dir
        step_dir = res_dpath / status
        if step_dir.exists():
            if self.ignore_exists:
                #warnings.warn("sorted_path removed in %s" %res_dpath, UserWarning)
                shutil.rmtree(step_dir)
                step_dir.mkdir()
            else:
                # TODO: How to deal with previous results?
                pass
        else:
            step_dir.mkdir()
        
        return step_dir
    
    @abstractmethod
    def _prior_create(self, input_params: dict, *args, **kwargs):
        """ some codes before creating exploratiosn of systems
            parse actions for this exploration from dict params
        """
        # - create
        # parse in subclasses

        # - collect
        collect_params_ = copy.deepcopy(self.collection_params)
        collect_params_.update(input_params.get("collect", None))
        self.collection_params = collect_params_

        # - select
        select_params = input_params.get("select", None)
        if select_params is not None:
            selector = create_selector(select_params, directory=Path.cwd())
        else:
            selector = None

        return selector

    @abstractmethod 
    def _single_create(self, res_dpath, frames, actions, *args, **kwargs):
        """ some codes run explorations
        """

        return

    @abstractmethod 
    def _single_collect(self, res_dpath, frames, actions, selector, *args, **kwargs):
        """ some codes run explorations
        """

        return
    
    def _check_create_status(self, res_dpath, *args, **kwargs):
        """ check whether create is finished
            and should perform collect or select
        """
        # TODO: store in a metadata file?
        status = "create"
        if (res_dpath/"CREATE_FINISHED").exists():
            status = "collect"
        if (res_dpath/"COLLECT_FINISHED").exists():
            status = "finished"

        return status
    
    def _post_create(self, *args, **kwargs):
        """ some codes after creating exploratiosn of systems
            collect data
        """

        return
    
    def _read_structure(self, slabel):
        """ read initial structures of a single system
            or generate structures from initial configurations
        """
        # - read structure
        system_dict = self.init_systems.get(slabel, None) # system name
        if system_dict is None:
            raise RuntimeError(f"Find unexpected system {system_dict}.")
        
        print("reading initial structures...")
        
        # - read structures
        # the expedition can start with different initial configurations
        init_frame_path = self.step_dpath / "init.xyz" 
        if init_frame_path.exists():
            print("read existed structure file...")
            frames = read(init_frame_path, ":")
        else:
            print("try to use generator...")
            stru_path = system_dict.get("structure", None)
            gen_params = system_dict.get("generator", None)
            if (stru_path is None and gen_params is not None): 
                from GDPy.builder.interface import create_generator
                generator = create_generator(gen_params)
                generator.directory = self.step_dpath
                frames = generator.run(system_dict.get("size", 1))
                #if self.name != "gs": # global search
                #    # global search expedition doesnt need initial structures
                #    raise RuntimeError(f"{self.name} needs initial structures of {slabel}")
            elif (stru_path is not None and gen_params is None):
                indices = system_dict.get("index", ":")
                frames = read(stru_path, indices)
            else:
                raise RuntimeError("Use either structure or generation...")
        
            write(self.step_dpath/"init.xyz", frames)
        
        print("number of initial structures: ", len(frames)) # TODO: use logging

        sys_cons_text = system_dict.get("constraint", None) # some action may need constraint info

        return frames, sys_cons_text

    def icalc(self, exp_name, working_directory, skipped_systems=[]):
        """calculate configurations with reference method"""
        exp_dict = self.explorations[exp_name]

        # - some parameters
        self.label_params.update(
            exp_dict.get("label", {})
        )

        # - create a calculation machine (vasp, ...)
        if self.ref_manager is None:
            raise RuntimeError("Ref Manager does not exist...")
        else:
            #calc_machine = self.ref_manager.create_machine(
            #    calc_dict
            #)
            # TODO: too complex here!!!
            calc_machine = self.ref_manager.create_machine(
                self.ref_manager.calc, 
                self.machine_dict.get(exp_dict["label"].get("mach_name", "mach1"), None)
            )

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
                res_dpath = working_directory / exp_name / slabel

                # - read collected/selected frames
                sorted_path = res_dpath / "select"
                if not sorted_path.exists():
                    sorted_path = res_dpath / "collect"
                    if not sorted_path.exists():
                        print(f"No candidates to calculate in {str(res_dpath)}")
                        continue

                found_files = {}
                for tag_name in self.collection_params["selection_tags"]:
                    # - find all selected files
                    # or find final selected that is produced by a composed selector
                    # TODO: if no selected were applied?
                    if sorted_path.name == "collect":
                        xyzfiles = list(sorted_path.glob(f"{tag_name}*.xyz"))
                    if sorted_path.name == "select":
                        xyzfiles = list(sorted_path.glob(f"{tag_name}*-selection*.xyz"))
                    #print(tag_name, xyzfiles)
                    nfiles = len(xyzfiles)
                    if nfiles > 0:
                        if nfiles == 1:
                            final_selected_path = xyzfiles[0]
                        else:
                            # assert files have selection order
                            xyzfiles = sorted(xyzfiles, key=lambda x:int(x.name.split(".")[0].split("-")[-1]))
                            final_selected_path = xyzfiles[-1]
                        if final_selected_path.stat().st_size > 0:
                            print(f"found selected structure file {str(final_selected_path)}")
                            print("nframes: ", len(read(final_selected_path, ":")))
                            # - create input files
                            fp_path = prefix / slabel / tag_name
                            found_files[tag_name] = [fp_path, final_selected_path]
                        else:
                            print(f"Cant find selected structure file with tag {tag_name}")
                    else:
                        print(f"Cant find selected structure file with tag {tag_name}")
                    
                # - run preparation
                for tag_name, (fp_path, final_selected_path) in found_files.items():
                    self._prepare_calc_dir(
                        calc_machine,
                        slabel, fp_path, 
                        final_selected_path
                    )

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
            # - update system-wise parameters
            extra_params = dict(
                system = slabel,
                # NOTE: kpts not for all calculators?
                kpts = self.init_systems[slabel].get("kpts", [1,1,1])
            )

            # - machine params
            user_commands = "gdp vasp work {} -in {}".format(
                str(final_selected_path.resolve()), (sorted_fp_path/"vasp_params.json").resolve()
            )

            # - create input files
            calc_machine._prepare_calculation(sorted_fp_path, extra_params, user_commands)
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

            xyzfile_path = database_path / ("-".join([slabel, self.name, sorted_fp_path.name, self.pot_manager.version]) + ".xyz")
            print(xyzfile_path)
            #exit()

            if xyzfile_path.exists():
                frames_exists = read(xyzfile_path, ":")
                nframes_exists = len(frames_exists)
                if nframes_exists == nframes_in:
                    print("  calculated structures have been stored...")
                    #return
                else:
                    print("  calculated structures may not finish...")
                frames = frames_exists
            else:
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

                write(database_path / xyzfile_path, frames)

            # - check parity
            if self.label_params["check_parity"]:
                from GDPy.data.operators import calc_and_compare_results, plot_comparasion

                # use loaded frames
                with CustomTimer(name="comparasion"):
                    figpath = sorted_fp_path / "cmp.png"
                    calc_name = self.pot_manager.calc.name.lower()
                    energies, forces = calc_and_compare_results(frames, self.pot_manager.calc)
                    plot_comparasion(calc_name, energies, forces, figpath)

        return

if __name__ == "__main__":
    pass