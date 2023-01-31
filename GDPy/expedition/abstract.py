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
import yaml

from ase import Atoms
from ase.io import read, write

from GDPy import config
from GDPy.utils.command import CustomTimer

from GDPy.selector import create_selector
from GDPy.computation.worker.drive import DriverBasedWorker


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

    def __init__(self, main_dict, potter: DriverBasedWorker, referee: DriverBasedWorker=None):
        """"""
        self._register_type_map(main_dict) # obtain type_list or type_map

        self.explorations = main_dict["explorations"]
        self.init_systems = main_dict["systems"]

        self._parse_general_params(main_dict)

        self.njobs = config.NJOBS

        # - potential and reference
        self.pot_worker = potter
        self.ref_worker = referee

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
            # NOTE: check if config has been changed
            exp_params = self.explorations[exp_name]
            config_fpath = exp_directory/"config.yaml"
            if config_fpath.exists():
                with open(config_fpath, "r") as fopen:
                    stored_params = yaml.safe_load(fopen)
                differ_keys = []
                for k, v in exp_params.items():
                    stored_v = stored_params.get(k, None)
                    if v != stored_v:
                        differ_keys.append(k)
                if differ_keys:
                    self.logger.info(f"Found inconsistent params for {exp_name}...")
                    self.logger.info(differ_keys)
                    continue
            else:
                with open(config_fpath, "w") as fopen:
                    yaml.safe_dump(exp_params, fopen, indent=2)
            # ---
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
            # NOTE: three types of params are allowed
            #       1. a list of selectors for all type of candidates
            #       2. Mapping[str,List[selector]] key for one type of candidates
            #          conv - local minima, traj - trajectories
            #          default - for other not specificed candidate group
            #       3.
            selections = {}
            if isinstance(select_params, list):
                selector = create_selector(select_params, directory=Path.cwd(), pot_worker=self.pot_worker)
                selections["default"] = selector
            else: # Mapping
                assert isinstance(select_params, dict), "Parameters for select should either be a list or a dict."
                for k, v in select_params.items():
                    selections[k] = create_selector(v, directory=Path.cwd(), pot_worker=self.pot_worker)
        else:
            selections = None
        actions["selector"] = selections

        # - label

        return actions

    def _irun(self, exp_name: str, exp_directory: Union[str,Path]) -> NoReturn:
        """ create expedition tasks and gather results
        """
        # - a few info
        exp_dict = self.explorations[exp_name]
        included_systems = exp_dict.get("systems", None)
        assert len(included_systems)>0, f"Expedition {self.name} needs at least one system."

        actions = self._prior_create(exp_dict) # NOTE: shared by different systems

        for slabel in included_systems:
            self.logger.info(f"\n\n===== Explore System {slabel} =====")

            # - prepare output directory
            res_dpath = exp_directory / slabel
            if not res_dpath.exists():
                res_dpath.mkdir(parents=True)

            # - read substrate
            self.step_dpath = self._make_step_dir(res_dpath, "init")
            # TODO: parse constraint in the structure reader?
            cons_text = self._read_structure(slabel, actions)

            init_frame_path = self.step_dpath / "init.xyz" 
            generator = actions["generator"]
            init_frames = generator.run(
                ran_size=self.init_systems[slabel].get("size", 1)
            )
            if not init_frame_path.exists():
                write(
                    init_frame_path, init_frames, columns=["symbols", "positions", "move_mask"]
                )
            else:
                # TODO: assert current init_frames is the same as the cached one
                pass

            # --- update cons text
            if isinstance(actions["driver"], list):
                for driver in actions["driver"]:
                    driver.run_params.update(constraint=cons_text)
            else:
                actions["driver"].run_params.update(constraint=cons_text)

            # - prepare data shared by different stages
            data = dict(
                init_frames = init_frames, # input candidates
                pot_frames = [], # created structures
                ref_frames = [] # structures need label
            )
            
            # - run exploration
            status = "create"
            self.logger.info(f"\n\n===== current status: {status}  =====")
            self.step_dpath = self._make_step_dir(res_dpath, status)
            is_created = self._single_create(res_dpath, actions, data)
            if not is_created:
                self.logger.info("Creation is not finished...")
                continue
            else:
                self.logger.info("Creation is FINISHED.")

            status = "collect"
            self.logger.info(f"\n\n===== current status: {status}  =====")
            self.step_dpath = self._make_step_dir(res_dpath, status)
            is_collected = self._single_collect(res_dpath, actions, data)
            if not is_collected:
                self.logger.info("Collect is not finished...")
                continue
            else:
                self.logger.info("Collect is FINISHED.")

            status = "select"
            self.logger.info(f"\n\n===== current status: {status}  =====")
            self.step_dpath = self._make_step_dir(res_dpath, status)
            is_selected = self._single_select(res_dpath, actions, data)
            if not is_selected:
                self.logger.info("Select is not finished...")
                continue
            else:
                self.logger.info("Select is FINISHED.")

            status = "label"
            self.logger.info(f"\n\n===== current status: {status}  =====")
            self.step_dpath = self._make_step_dir(res_dpath, status)
            self._single_label(res_dpath, actions, data)

            # TODO: benchmark...

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
    def _single_create(self, res_dpath, actions, data, *args, **kwargs):
        """ some codes run explorations
        """

        return

    @abstractmethod 
    def _single_collect(self, res_dpath, actions, data, *args, **kwargs):
        """ some codes run explorations
        """

        return

    def _single_select(self, res_dpath, actions, data, *args, **kwargs):
        """Select representative structures from candidates for further processing.

        Different selection protocols can be set for different type of structures.

        """
        is_selected = True
        # - select
        selections = actions["selector"] # NOTE: default is invariant
        if selections:
            # - find frames...
            valid_keys = []
            for k in data.keys():
                if k.startswith("pot_frames_"):
                    valid_keys.append(k)
            if not valid_keys:
                valid_keys.append("pot_frames")
            
            # - check if have specifc selectors
            default_selector = selections.get("default", None)
            
            # - run selection
            for k in valid_keys:
                tag_name = k.split("_")[-1]
                if tag_name == "frames":
                    tag_name = "mixed"

                current_selector = selections.get(tag_name, None)
                if current_selector is None:
                    current_selector = default_selector
                if current_selector is None:
                    self.logger.info(f"----- No Selection for {tag_name} -----")
                else:
                    self.logger.info(f"----- Selection {current_selector.name} for {tag_name} -----")

                cur_dpath = res_dpath/"select"/tag_name
                if not cur_dpath.exists():
                    cur_dpath.mkdir()
                current_selector.directory = cur_dpath
                current_selector.logger = self.logger

                frames = data[k]
                # BUG: input frames are inconsitent due to source info?
                selected_frames = current_selector.select(frames)
                if selected_frames:
                    data["selected_frames_"+tag_name] = selected_frames
                    if not cur_dpath.exists():
                        self.logger.info("save structures to xyz file...")
                        write(cur_dpath/"selected_frames.xyz", selected_frames)
                else:
                    data["selected_frames_"+tag_name] = []
                    self.logger.info(f"No candidates were found for {tag_name}.")
        else:
            self.logger.info("No selector available...")

        return is_selected
    
    def _post_create(self, *args, **kwargs):
        """ some codes after creating exploratiosn of systems
            collect data
        """

        return
    
    def _read_structure(self, slabel, actions: dict):
        """ read initial structures of a single system
            or generate structures from initial configurations
        """
        # - read structure
        system_dict = self.init_systems.get(slabel, None) # system name
        if system_dict is None:
            raise RuntimeError(f"Find unexpected system {system_dict}.")
        
        self.logger.info("reading initial structures...")
        
        # - read structures
        from GDPy.builder import create_generator
        # the expedition can start with different initial configurations
        #if init_frame_path.exists():
        #    self.logger.info("read cached structure file...")
        #    #from GDPy.builder.direct import DirectGenerator
        #    #generator = DirectGenerator(init_frame_path)
        #    gen_params = init_frame_path
        self.logger.info("try to use generator...")
        stru_path = system_dict.get("structure", None)
        gen_params = system_dict.get("generator", None)
        if (stru_path is not None and gen_params is None): 
            #from GDPy.builder import create_generator
            #generator = create_generator(gen_params)
            #generator.directory = self.step_dpath
            #frames = generator.run(system_dict.get("size", 1))
            gen_params = Path(stru_path).resolve()
        elif (stru_path is None and gen_params is not None):
            #indices = system_dict.get("index", ":")
            #from GDPy.builder.direct import DirectGenerator
            #generator = DirectGenerator(stru_path, indices)
            #generator = DirectGenerator(stru_path)
            gen_params = gen_params
        else:
            raise RuntimeError("Use either structure or generation...")
        generator = create_generator(gen_params)
        generator.directory = self.step_dpath
        generator.logger = self.logger

        # NOTE: population-based method (GA) needs a generator as a dict in its
        #       input file...
        actions["generator"] = generator
        
        sys_cons_text = system_dict.get("constraint", None) # some action may need constraint info

        return sys_cons_text
    
    def _single_label(self, res_dpath, actions, data, *args, **kwargs):
        """"""
        if not self.ref_worker:
            self.logger.info("Reference worker is not set properly.")
            return

        # - find frames from data
        valid_keys = []
        for k in data.keys():
            if k.startswith("selected_frames"):
                valid_keys.append(k)
        if not valid_keys:
            valid_keys.append("pot_frames") # used collected one

        # - run preparation
        for key in valid_keys:
            frames = data[key]
            tag_name = key.split("_")[-1]
            if tag_name == "frames":
                tag_name = "mixed"
            fp_path = res_dpath/"label"/tag_name
            nframes = len(frames)
            self.logger.info(f"tag: {tag_name} nframes: {nframes}")
            # NOTE: remove existed wdir info
            #       since previous calculations may add them
            for a in frames:
                _ = a.info.pop("wdir", None)
            if nframes > 0:
                self._prepare_calc_dir(
                    res_dpath.name, fp_path, frames
                )

        return
    
    def _prepare_calc_dir(
        self, slabel, sorted_fp_path, frames_in
    ):
        """ prepare calculation dir
            currently, only vasp is supported
        """
        # NOTE: convert into atoms' calculator into spc
        #       some torch/tensorflow caclulators cant be pickled when copy
        from ase.calculators.singlepoint import SinglePointCalculator
        for atoms in frames_in:
            results = dict(
                energy = atoms.get_potential_energy(),
                forces = atoms.get_forces().copy()
            )
            calc = SinglePointCalculator(atoms, **results)
            atoms.calc = calc

        # - check target calculation structures
        nframes_in = len(frames_in)

        # - update some specific params of worker
        worker = self.ref_worker
        worker.logger = self.logger
        #worker._submit = False
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
        worker.inspect(resubmit=True) # BUG: input frames are inconsitent due to source info?
        if worker.get_number_of_running_jobs() > 0:
            self.logger.info("jobs are running...")
        else:
            # TODO: write structues when all are finished, not append
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