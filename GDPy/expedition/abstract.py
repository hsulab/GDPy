#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod
from typing import Union, Callable, Counter, Union, List
from pathlib import Path

import time
import json
import warnings

from joblib import Parallel, delayed

from ase import Atoms
from ase.io import read, write

from GDPy import config
from GDPy.machine.machine import SlurmMachine
from GDPy.utils.data import vasp_creator, vasp_collector


""" abstract class for expedition methods
    each one has following procedures:
        creation
            system-name-based-dir
        collection+selection
            sorted-dir
        calculation (includes harvest)
            fp-dir
    a calculator/worker needs defined to drive the exploration
"""


class AbstractExplorer(ABC):

    # general parameters
    general_params = dict(
        ignore_exists = False
    )

    creation_params = dict(

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

        # some parameters
        calc_dict = exp_dict["calculation"]
        machine_dict = calc_dict["machine"]

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
                            slabel, fp_path, calc_dict, machine_dict, 
                            final_selected_path
                        )
                else:
                    print(f"No candidates to calculate in {str(name_path)}")

        return
    
    def _prepare_calc_dir(
        self, slabel, sorted_fp_path, calc_dict, machine_dict,
        final_selected_path
    ):
        """"""
        # - create input file TODO: move this to vasp part
        if not sorted_fp_path.exists():
            sorted_fp_path.mkdir(parents=True)
            # -- update params with systemwise info
            calc_params = calc_dict.copy()
            for k in calc_params.keys():
                if calc_params[k] == "system":
                    calc_params[k] = self.init_systems[slabel][k]
            # -- create params file
            with open(sorted_fp_path/"vasp_params.json", "w") as fopen:
                json.dump(calc_params, fopen, indent=4)
            # -- create job script
            machine_params = machine_dict.copy()
            machine_params["job-name"] = slabel+"-fp"

            # TODO: mpirun or mpiexec, move this part to machine object
            command = calc_params["command"]
            if command.strip().startswith("mpirun") or command.strip().startswith("mpiexec"):
                ntasks = command.split()[2]
            else:
                ntasks = 1
            # TODO: number of nodes?
            machine_params.update(**{"nodes": "1-1", "ntasks": ntasks, "cpus-per-task": 1, "mem-per-cpu": "4G"})

            machine = SlurmMachine(**machine_params)
            #"gdp vasp work ../C3O3Pt36.xyz -in ../vasp_params.json"
            machine.user_commands = "gdp vasp work {} -in {}".format(
                str(final_selected_path.resolve()), (sorted_fp_path/"vasp_params.json").resolve()
            )
            machine.write(sorted_fp_path/"vasp.slurm")
            # -- submit?
        else:
            # TODO: move harvest function here?
            print(f"{sorted_fp_path} already exists.")

        return
    
    def iharvest(self, exp_name, working_directory: Union[str, Path]):
        """harvest all vasp results"""
        # TODO: replace this by a object
        # run over directories and check
        main_dir = Path(working_directory) / (exp_name + "-fp")
        vasp_main_dirs = []
        for p in main_dir.iterdir():
            calc_file = p / "calculated_0.xyz"
            if p.is_dir() and calc_file.exists():
                vasp_main_dirs.append(p)
        print(vasp_main_dirs)

        # TODO: optional parameters
        pot_gen = Path.cwd().name
        pattern = "vasp_0_*"
        njobs = 4
        vaspfile, indices = "vasprun.xml", "-1:"

        for d in vasp_main_dirs:
            print("\n===== =====")
            vasp_dirs = []
            for p in d.parent.glob(d.name+'*'):
                if p.is_dir():
                    vasp_dirs.extend(vasp_collector.find_vasp_dirs(p, pattern))
            print('total vasp dirs: %d' %(len(vasp_dirs)))

            print("sorted by last integer number...")
            vasp_dirs_sorted = sorted(
                vasp_dirs, key=lambda k: int(k.name.split('_')[-1])
            ) # sort by name

            # check number of frames equal output?
            input_xyz = []
            for p in d.iterdir():
                if p.name.endswith("-sel.xyz"):
                    input_xyz.append(p)
                if p.name.endswith("_ALL.xyz"):
                    input_xyz.append(p)
            if len(input_xyz) == 1:
                input_xyz = input_xyz[0]
            else:
                raise ValueError(d, " has both sel and ALL xyz file...")
            nframes_input = len(read(input_xyz, ":"))

            atoms = read(input_xyz, "0")
            c = Counter(atoms.get_chemical_symbols())
            sys_name_list = []
            for s in self.type_list:
                sys_name_list.append(s)
                num = c.get(s, 0)
                sys_name_list.append(str(num))
            sys_name = "".join(sys_name_list)
            out_name = self.main_database / sys_name / (d.name + "-" + pot_gen + ".xyz")
            if out_name.exists():
                nframes_out = len(read(out_name, ":"))
                if nframes_input == nframes_out:
                    print(d, "already has been harvested...")
                    continue

            # start harvest
            st = time.time()
            print("using num of jobs: ", njobs)
            cur_frames = Parallel(n_jobs=njobs)(delayed(vasp_collector.extract_atoms)(p, vaspfile, indices) for p in vasp_dirs_sorted)
            if isinstance(cur_frames, Atoms):
                cur_frames = [cur_frames]
            frames = []
            for f in cur_frames:
                frames.extend(f) # merge all frames

            et = time.time()
            print("cost time: ", et-st)

            # move structures to data path
            if len(frames) > 0:
                print("Number of frames: ", len(frames))
                write(out_name, frames)
            else:
                print("No frames...")

        return


if __name__ == "__main__":
    pass