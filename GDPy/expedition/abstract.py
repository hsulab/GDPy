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
import logging
import warnings

from ase import Atoms
from ase.io import read, write

from GDPy import config
from GDPy.utils.command import CustomTimer

from GDPy.selector import create_selector


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


class AbstractExpedition(ABC):

    # name of expedition
    name = "expedition"

    parameters = dict()

    restart = True # for the logger

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
        check_parity = False
    )

    # system-specific info
    type_map = {}
    type_list = []

    def __init__(self, main_dict, potter, referee=None):
        """"""
        self._register_type_map(main_dict) # obtain type_list or type_map

        self.explorations = main_dict["explorations"]
        self.init_systems = main_dict["systems"]

        self._parse_general_params(main_dict)

        self.njobs = config.NJOBS

        # - potential and reference
        self.pot_worker = potter
        self.ref_worker = referee
        print("pot_worker: ", self.pot_worker)

        return

    def _init_logger(self, working_directory):
        """"""
        self.logger = logging.getLogger(__name__)

        log_level = logging.INFO

        self.logger.setLevel(log_level)

        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        working_directory = Path(working_directory)
        log_fpath = working_directory / (self.name+".out")

        if self.restart:
            fh = logging.FileHandler(filename=log_fpath, mode="a")
        else:
            fh = logging.FileHandler(filename=log_fpath, mode="w")

        fh.setLevel(log_level)
        #fh.setFormatter(formatter)

        ch = logging.StreamHandler()
        ch.setLevel(log_level)
        #ch.setFormatter(formatter)

        self.logger.addHandler(ch)
        self.logger.addHandler(fh)

        # begin!
        self.logger.info(
            "\nStart at %s\n", 
            time.asctime( time.localtime(time.time()) )
        )

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
            self.main_database = Path(main_database).resolve()

        return
    
    def parse_specorder(self, composition: dict):
        """ determine species order from current structure
            since many systems with different compositions 
            may be explored at the same time
        """
        type_list = []
        for sym, num in composition.items():
            self.logger.info(sym, num)
            if num != 0:
                type_list.append(sym)

        return type_list
    
    def run(
        self, 
        working_directory: Union[str, Path]
    ): 
        """create for all explorations"""
        working_directory = Path(working_directory)
        if not working_directory.exists():
            working_directory.mkdir(parents=True)
        for exp_name in self.explorations.keys():
            exp_directory = working_directory / exp_name
            if not exp_directory.exists():
                exp_directory.mkdir(parents=True)
            self._init_logger(exp_directory)
            self._irun(exp_name, exp_directory)

        return

    @abstractmethod
    def _prior_create(self, input_params: dict, *args, **kwargs) -> dict:
        """ some codes before creating exploratiosn of systems
            parse actions for this exploration from dict params
        """
        actions = {}

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
        actions["selector"] = selector

        # - label

        return actions

    def _irun(self, exp_name: str, exp_directory: Union[str,Path]) -> NoReturn:
        """ create expedition tasks and gather results
        """
        # - a few info
        exp_dict = self.explorations[exp_name]
        included_systems = exp_dict.get("systems", None)
        assert len(included_systems)>0, f"Expedition {self.name} needs at least one system."

        actions = self._prior_create(exp_dict)

        for slabel in included_systems:
            self.logger.info(f"\n\n===== Explore System {slabel} =====")

            # - prepare output directory
            res_dpath = exp_directory / slabel
            if not res_dpath.exists():
                res_dpath.mkdir(parents=True)

            # - read substrate
            self.step_dpath = self._make_step_dir(res_dpath, "init")
            generator, cons_text = self._read_structure(slabel)
            actions["generator"] = generator

            # --- update cons text
            # TODO: need a unified interface here...
            if self.name == "rxn":
                actions["reaction"].constraint = cons_text
            else:
                if isinstance(actions["driver"], list):
                    for driver in actions["driver"]:
                        driver.run_params.update(constraint=cons_text)
                else:
                    actions["driver"].run_params.update(constraint=cons_text)
            
            # - run exploration
            status = "create"
            self.logger.info(f"\n\n===== current status: {status}  =====")
            self.step_dpath = self._make_step_dir(res_dpath, status)
            is_created = self._single_create(
                res_dpath, actions, ran_size=self.init_systems[slabel].get("size", 1)
            )
            if not is_created:
                self.logger.info("Creation is not finished...")
                continue
            else:
                self.logger.info("Creation is FINISHED.")

            status = "collect"
            self.logger.info(f"\n\n===== current status: {status}  =====")
            self.step_dpath = self._make_step_dir(res_dpath, status)
            is_collected = self._single_collect(res_dpath, actions)
            if not is_collected:
                self.logger.info("Collect/Select is not finished...")
                continue
            else:
                self.logger.info("Collect/Select is FINISHED.")

            status = "label"
            self.logger.info(f"\n\n===== current status: {status}  =====")
            self.step_dpath = self._make_step_dir(res_dpath, status)
            self._single_label(res_dpath, actions)

        self._post_create()

        return
    
    def _make_step_dir(self, res_dpath: Path, status: str) -> Path:
        """"""
        if status not in ["init", "create", "collect", "select", "label", "train"]:
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
    def _single_create(self, res_dpath, actions, *args, **kwargs):
        """ some codes run explorations
        """

        return

    @abstractmethod 
    def _single_collect(self, res_dpath, actions, *args, **kwargs):
        """ some codes run explorations
        """

        return

    def _single_select(self, res_dpath, frames, actions, *args, **kwargs):
        """ some codes run explorations
        """
        is_selected = True
        # - select
        selector = actions["selector"]
        if selector:
            select_dpath = self._make_step_dir(res_dpath, "select")

            selector.prefix = "traj"
            selector.directory = select_dpath
            selector.logger = self.logger

            # TODO: check whether select is finished...
            selected_frames = selector.select(frames)

        return is_selected
    
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
    
    def _check_status(self, status, *args, **kwargs):
        """"""

        return
    
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
        
        self.logger.info("reading initial structures...")
        
        # - read structures
        # the expedition can start with different initial configurations
        init_frame_path = self.step_dpath / "init.xyz" 
        if init_frame_path.exists():
            self.logger.info("read cached structure file...")
            from GDPy.builder.direct import DirectGenerator
            frames = read(init_frame_path, ":")
            generator = DirectGenerator(frames, self.step_dpath)
        else:
            self.logger.info("try to use generator...")
            stru_path = system_dict.get("structure", None)
            gen_params = system_dict.get("generator", None)
            if (stru_path is None and gen_params is not None): 
                from GDPy.builder.interface import create_generator
                generator = create_generator(gen_params)
                generator.directory = self.step_dpath
                #frames = generator.run(system_dict.get("size", 1))
            elif (stru_path is not None and gen_params is None):
                indices = system_dict.get("index", ":")
                frames = read(stru_path, indices)
                from GDPy.builder.direct import DirectGenerator
                generator = DirectGenerator(frames, self.step_dpath)
            else:
                raise RuntimeError("Use either structure or generation...")
        
        sys_cons_text = system_dict.get("constraint", None) # some action may need constraint info

        return generator, sys_cons_text
    
    def _single_label(self, res_dpath, actions, *args, **kwargs):
        """"""
        assert self.ref_worker, "Reference worker is not set properly."

        # - read collected/selected frames
        sorted_path = res_dpath / "select"
        if not sorted_path.exists():
            sorted_path = res_dpath / "collect"
            if not sorted_path.exists():
                self.logger.info(f"No candidates to calculate in {str(res_dpath)}")
                return

        found_files = {}
        for tag_name in self.collection_params["selection_tags"]:
            # - find all selected files
            # or find final selected that is produced by a composed selector
            if sorted_path.name == "collect":
                xyzfiles = list(sorted_path.glob(f"{tag_name}*.xyz"))
            if sorted_path.name == "select":
                xyzfiles = list(sorted_path.glob(f"{tag_name}*-selection*.xyz"))
            #self.logger.info(tag_name, xyzfiles)
            nfiles = len(xyzfiles)
            if nfiles > 0:
                if nfiles == 1:
                    final_selected_path = xyzfiles[0]
                else:
                    # assert files have selection order
                    xyzfiles = sorted(xyzfiles, key=lambda x:int(x.name.split(".")[0].split("-")[-1]))
                    final_selected_path = xyzfiles[-1]
                if final_selected_path.stat().st_size > 0:
                    self.logger.info(f"found selected structure file {str(final_selected_path)}")
                    final_frames = read(final_selected_path, ":")
                    self.logger.info(f"nframes: {len(final_frames)}")
                    # - create input files
                    fp_path = res_dpath / "label" / tag_name
                    found_files[tag_name] = [fp_path, final_selected_path]
                else:
                    self.logger.info(f"Cant find selected structure file with tag {tag_name}")
            else:
                self.logger.info(f"Cant find selected structure file with tag {tag_name}")
            
        # - run preparation
        for tag_name, (fp_path, final_selected_path) in found_files.items():
            self._prepare_calc_dir(
                res_dpath.name, fp_path, 
                final_selected_path
            )

        return
    
    def _prepare_calc_dir(
        self, slabel, sorted_fp_path, final_selected_path
    ):
        """ prepare calculation dir
            currently, only vasp is supported
        """
        # - check target calculation structures
        frames_in = read(final_selected_path, ":")
        nframes_in = len(frames_in)

        # - update some specific params of worker
        worker = self.ref_worker
        worker.logger = self.logger
        worker._submit = True
        #worker.prefix = "_".join([slabel,"Worker"])
        worker.batchsize = nframes_in

        # - try to run and submit jobs
        system_dict = self.init_systems[slabel]
        for k in worker.driver.syswise_keys:
            v = system_dict.get(k, None)
            if v:
                worker.driver.init_params.update(**{k: v})
            
        worker.directory = sorted_fp_path
        worker.run(frames_in)

        # - try to harvest
        worker.inspect()
        if worker.get_number_of_running_jobs() > 0:
            self.logger.info("jobs are running...")
        else:
            calculated_fpath = worker.directory/"calculated.xyz"
            new_frames = worker.retrieve()
            if new_frames:
                write(calculated_fpath, new_frames, append=True)
                self.logger.info(f"nframes newly added: {len(new_frames)}")
                # - tag every data
                # TODO: add uuid in worker?
                for atoms in new_frames:
                    atoms.info["uuid"] = str(uuid.uuid1())
            assert len(worker._get_unretrieved_jobs()) == 0, "still have jobs not retrieved."

            calculated_frames = read(calculated_fpath, ":")
            
            # - try to copy caclualted structures to centre database
            # TODO: provide an unified interfac to all type of databases 
            # TODO: offer default xyzfile prefix
            composition = self.init_systems[slabel]["composition"] # if not provided, infer from generator
            sorted_composition = sorted(composition.items(), key=lambda x:x[0])
            comp_name = "".join([ k+str(v) for k,v in sorted_composition])
            database_path = self.main_database / (self.init_systems[slabel]["prefix"]+"-"+comp_name)

            if not database_path.exists():
                database_path.mkdir(parents=True)

            xyzfile_path = database_path / ("-".join([slabel, self.name, sorted_fp_path.name, worker.potter.version]) + ".xyz")
            self.logger.info(str(xyzfile_path))

            write(database_path / xyzfile_path, calculated_frames)
        
        return


if __name__ == "__main__":
    pass