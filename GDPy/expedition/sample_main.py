#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import time
import shutil
import json
import sys
import itertools
from typing import Counter, Union, List
import warnings
import pathlib
from joblib import Parallel, delayed
import numpy as np
import numpy.ma as ma

import dataclasses
from dataclasses import dataclass, field

from ase import Atoms
from ase.io import read, write
from ase.io.lammpsrun import read_lammps_dump_text
from ase.data import atomic_numbers, atomic_masses
from ase.constraints import constrained_indices, FixAtoms

from GDPy.calculator.ase_interface import AseInput
from GDPy.calculator.inputs import LammpsInput

from GDPy.machine.machine import SlurmMachine

from GDPy.utils.data import vasp_creator, vasp_collector
from GDPy.utils.command import parse_input_file, convert_indices

from GDPy.expedition.abstract import AbstractExplorer
from GDPy.selector.abstract import create_selector


@dataclasses.dataclass
class MDParams:        

    #unit = "ase"
    method: str = "md"
    md_style: str = "nvt" # nve, nvt, npt
    steps: int = 0 
    dump_period: int = 1 
    timestep: float = 2 # fs
    temp: float = 300 # Kelvin
    pres: float = -1 # bar

    # fix nvt/npt/nph
    Tdamp: float = 100 # fs
    Pdamp: float = 500 # fs

    def __post_init__(self):
        """ unit convertor
        """

        return

def create_dataclass_from_dict(dcls: dataclasses.dataclass, params: dict) -> List[dataclasses.dataclass]:
    """ create a series of dcls instances
    """
    # NOTE: onlt support one param by list
    # - find longest params
    plengths = []
    for k, v in params.items():
        if isinstance(v, list):
            n = len(v)
        else: # int, float, string
            n = 1
        plengths.append((k,n))
    plengths = sorted(plengths, key=lambda x:x[1])
    # NOTE: check only has one list params
    assert sum([p[1] > 1 for p in plengths]) <= 1, "only accept one param as list."

    # - convert to dataclass
    dcls_list = []
    maxname, maxlength = plengths[-1]
    for i in range(maxlength):
        cur_params = {}
        for k, n in plengths:
            if n > 1:
                v = params[k][i]
            else:
                v = params[k]
            cur_params[k] = v
        dcls_list.append(dcls(**cur_params))

    return dcls_list


class MDBasedExpedition(AbstractExplorer):

    """
    Exploration Strategies
        1. random structure sampling
        2. small pertubations on given structures
        3. molecular dynamics with surrogate potential
        4. molecular dynamics with uncertainty-aware potential
    
    Initial Systems
        initial structures must be manually prepared
    
    Units
        fs, eV, eV/AA
    
    Workflow
        create+/run-collect+/select-calculate+/harvest
    """

    method = "MD" # nve, nvt, npt
    backend = "lammps" # run MD in lammps, maybe ASE

    # TODO: !!!!
    # check whether submit jobs
    # check system symbols with type list
    # check lost atoms when collecting
    # check overwrite during collect and select and calc
    # check different structures input format in collect

    supported_potentials = ["reax", "deepmd", "eann"]
    supported_procedures = ["create", "collect", "select", "calculate"]

    default_params = {
        "collect": {
            "deviation": None
        }
    }

    # general parameters
    general_params = dict(
        ignore_exists = False
    )

    def __init__(self, pm, main_dict: dict):
        """"""
        self.pot_manager = pm
        self._register_type_map(main_dict)
        #assert self.pot_manager.type_map == self.type_map, 'type map should be consistent'

        self.explorations = main_dict['explorations']
        self.init_systems = main_dict['systems']

        self._parse_general_params(main_dict)

        # for job prefix
        self.job_prefix = ""


        return
    
    def _parse_dyn_params(self, exp_dict: dict):
        """ create a list of workers based on dyn params
        """
        dyn_params = exp_dict["dynamics"]
        #print(dyn_params)

        backend = dyn_params.pop("backend", None)

        #for m in itertools.zip_longest():
        #    return
        p = MDParams(**dyn_params)
        dcls_list = create_dataclass_from_dict(MDParams, dyn_params)

        workers = []
        for p in dcls_list:
            p_ = dataclasses.asdict(p)
            p_.update(backend=backend)
            worker = self.pot_manager.create_worker(p_)
            workers.append(worker)

        return workers

    def icreate(self, exp_name, working_directory):
        """create for each exploration"""
        # - a few info
        exp_dict = self.explorations[exp_name]
        job_script = exp_dict.get('jobscript', None)
        included_systems = exp_dict.get('systems', None)

        # - check info
        if included_systems is not None:
            # NOTE: create a list of workers
            workers = self._parse_dyn_params(exp_dict)

            # loop over systems
            for slabel in included_systems:
                # - result path
                name_path = working_directory / exp_name / slabel
                if not name_path.exists():
                    name_path.mkdir(parents=True)
                else:
                    # TODO: check ignore_exists
                    pass
                # NOTE: since multiple explorations are applied to one system, 
                # a metedata file shall be created to log the parameters
                pkeys = workers[0].dynrun_params.keys()
                content = "#" + " ".join(pkeys) + "\n"
                for w in workers:
                    c = [w.dynrun_params[k] for k in pkeys]
                    content += " ".join([str(x) for x in c]) + "\n"
                with open(name_path / "metadata.txt", "w") as fopen:
                    fopen.write(content)
                
                # - parse structure and composition
                system_dict = self.init_systems.get(slabel, None) # system name
                if system_dict is None:
                    raise ValueError(f"Find unexpected system {system_dict}.")
                scomp = system_dict['composition'] # system composition
                atypes = []
                for atype, number in scomp.items():
                    if number > 0:
                        atypes.append(atype)
                sys_cons_text = system_dict.get('constraint', None)

                # - read structures
                # the expedition can start with different initial configurations
                stru_path = system_dict["structure"]
                frames = read(stru_path, ":")

                # - run over systems
                for iframe, atoms in enumerate(frames):
                    name = atoms.info.get("name", "f"+str(iframe))
                    # TODO: check if atoms have constraint
                    cons_indices = constrained_indices(atoms, only_include=FixAtoms) # array
                    if cons_indices.size > 0:
                        # convert to lammps convention
                        cons_indices += 1
                        cons_text = convert_indices(cons_indices.tolist())
                    else:
                        cons_text = sys_cons_text
                    print("cons_text: ", cons_text)

                    work_path = name_path / name
                    print(work_path)

                    # - run simulation
                    for iw, worker in enumerate(workers):
                        worker.set_output_path(work_path/("w"+str(iw)))
                        # TODO: run directly or attach a machine
                        new_atoms = worker.run(atoms, constraint=cons_text)
                        print(new_atoms)
                        print(new_atoms.get_potential_energy())

        return
    
    def icollect(self, exp_name, working_directory, skipped_systems=[]):
        """collect data from single calculation"""
        exp_dict = self.explorations[exp_name]
        # - create a selector
        # TODO: use function from selector
        selection_params = exp_dict.get("selection", None)
        selectors = create_selector(selection_params)

        # - run over systems
        included_systems = exp_dict.get('systems', None)
        if included_systems is not None:
            # NOTE: create a list of workers
            workers = self._parse_dyn_params(exp_dict)

            # - loop over systems
            for slabel in included_systems:
                sys_frames = [] # NOTE: all frames
                # TODO: make this into system
                if slabel in skipped_systems:
                    continue
                # - result path
                name_path = working_directory / exp_name / slabel

                # - read structures
                # the expedition can start with different initial configurations
                system_dict = self.init_systems[slabel] # system name
                stru_path = system_dict["structure"]
                frames = read(stru_path, ":")

                # - parse structure and composition
                system_dict = self.init_systems.get(slabel, None) # system name
                if system_dict is None:
                    raise ValueError(f"Find unexpected system {system_dict}.")
                scomp = system_dict['composition'] # system composition
                atypes = []
                for atype, number in scomp.items():
                    if number > 0:
                        atypes.append(atype)
                sys_cons_text = system_dict.get('constraint', None)

                # - run over systems
                for iframe, atoms in enumerate(frames):
                    name = atoms.info.get("name", "f"+str(iframe))
                    # TODO: check if atoms have constraint
                    cons_indices = constrained_indices(atoms, only_include=FixAtoms) # array
                    if cons_indices.size > 0:
                        # convert to lammps convention
                        cons_indices += 1
                        cons_text = convert_indices(cons_indices.tolist())
                    else:
                        cons_text = sys_cons_text
                    print("cons_text: ", cons_text)

                    work_path = name_path / name
                    print(work_path)

                    # - run simulation
                    for iw, worker in enumerate(workers):
                        worker.set_output_path(work_path/("w"+str(iw)))
                        # TODO: run directly or attach a machine
                        traj_frames = worker._read_trajectory(atoms)
                        #write("xxx.xyz", traj_frames)
                        sys_frames.extend(traj_frames)
            
                # - systemwise selection
                if selectors is not None:
                    # -- create dir
                    sorted_path = name_path / "sorted"
                    if sorted_path.exists():
                        if self.ignore_exists:
                            warnings.warn("sorted_path removed in %s" %name_path, UserWarning)
                            shutil.rmtree(sorted_path)
                            sorted_path.mkdir()
                        else:
                            warnings.warn("sorted_path exists in %s" %name_path, UserWarning)
                            continue
                    else:
                        sorted_path.mkdir()
                    # -- perform selections
                    cur_frames = sys_frames
                    for isele, selector in enumerate(selectors):
                        # TODO: add info to selected frames
                        print(f"--- Selection {isele} Method {selector.name}---")
                        print("ncandidates: ", len(cur_frames))
                        cur_frames = selector.select(cur_frames)
                        print("nselected: ", len(cur_frames))
                    
                        write(sorted_path/f"{selector.name}-selected-{isele}.xyz", cur_frames)

        return
    
    def icalc(self, exp_name, working_directory, skipped_systems=[]):
        """calculate configurations with reference method"""
        exp_dict = self.explorations[exp_name]

        # some parameters
        calc_dict = exp_dict["calculation"]
        machine_dict = calc_dict["machine"]
        nstructures = calc_dict.get("nstructures", 100000) # number of structures in each calculation dirs

        # - create fp main dir
        prefix = working_directory / (exp_name + "-fp")
        if prefix.exists():
            warnings.warn("fp directory exists...", UserWarning)
        else:
            prefix.mkdir(parents=True)

        # - run over systems
        included_systems = exp_dict.get('systems', None)
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
                sorted_path = name_path / "sorted"
                if sorted_path.exists():
                    # - find all selected files
                    # TODO: if no selected were applied?
                    xyzfiles = list(sorted_path.glob("*.xyz"))
                    xyzfiles = sorted(xyzfiles, key=lambda x:int(x.name.split(".")[0].split("-")[-1]))
                    # - create input file
                    sorted_fp_path = prefix / slabel
                    if not sorted_fp_path.exists():
                        sorted_fp_path.mkdir()
                        # -- update params with systemwise info
                        calc_params = calc_dict.copy()
                        for k in calc_params.keys():
                            if calc_params[k] == "system":
                                calc_params[k] = self.init_systems[slabel][k]
                        with open(sorted_fp_path/"vasp_params.json", "w") as fopen:
                            json.dump(calc_params, fopen, indent=4)
                        # -- create job script
                        machine_params = machine_dict.copy()
                        machine_params["job-name"] = slabel+"-fp"

                        # TODO: mpirun or mpiexec
                        command = calc_params["command"]
                        if command.strip().startswith("mpirun"):
                            ntasks = command.split()[2]
                        else:
                            ntasks = 1
                        machine_params.update(**{"nodes": "1-1", "ntasks": ntasks, "cpus-per-task": 1, "mem-per-cpu": "4G"})

                        machine = SlurmMachine(**machine_params)
                        machine.user_commands = "python ~/repository/GDPy/GDPy/calculator/vasp.py {} -p {}".format(
                            str(xyzfiles[-1].resolve()), (sorted_fp_path/"vasp_params.json").resolve()
                        )
                        machine.write(sorted_fp_path/"vasp.slurm")
                    else:
                        print(f"{sorted_fp_path} already exists.")
                else:
                    print(f"No candidates to calculate in {str(name_path)}")

        return
    
    def iharvest(self, exp_name, working_directory: Union[str, pathlib.Path]):
        """harvest all vasp results"""
        # run over directories and check
        main_dir = pathlib.Path(working_directory) / (exp_name + "-fp")
        vasp_main_dirs = []
        for p in main_dir.iterdir():
            calc_file = p / "calculated_0.xyz"
            if p.is_dir() and calc_file.exists():
                vasp_main_dirs.append(p)
        print(vasp_main_dirs)

        # TODO: optional parameters
        pot_gen = pathlib.Path.cwd().name
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
    

def run_exploration(pot_manager, exp_json, chosen_step, global_params = None):
    # create exploration
    #with open(exp_json, 'r') as fopen:
    #    exp_dict = json.load(fopen)
    exp_dict = parse_input_file(exp_json)

    method = exp_dict.get("method", "MD")
    if method == "MD":
        scout = MDBasedExpedition(pot_manager, exp_dict)
    elif method == "GA":
        from GDPy.expedition.structure_exploration import RandomExplorer
        scout = RandomExplorer(pot_manager, exp_dict)
    else:
        raise ValueError(f"Unknown method {method}")


    # adjust global params
    print("optional params ", global_params)
    if global_params is not None:
        assert len(global_params)%2 == 0, "optional params must be key-pair"
        for first in range(0, len(global_params), 2):
            print(global_params[first], " -> ", global_params[first+1])
            scout.default_params[chosen_step][global_params[first]] = eval(global_params[first+1])

    # compute
    op_name = "i" + chosen_step
    assert isinstance(op_name, str), "op_nam must be a string"
    op = getattr(scout, op_name, None)
    if op is not None:
        scout.run(op, "./")
    else:
        raise ValueError("Wrong chosen step %s..." %op_name)

    return


if __name__ == '__main__':
    import json
    with open('/users/40247882/repository/GDPy/templates/inputs/main.json', 'r') as fopen:
        main_dict = json.load(fopen)
    
    exp_dict = main_dict['explorations']['reax-surface-diffusion']
    md_prefix = pathlib.Path('/users/40247882/projects/oxides/gdp-main/reax-metad')
    init_systems = main_dict['systems']
    type_map = {'O': 0, 'Pt': 1}

    icollect_data(exp_dict, md_prefix, init_systems, type_map)
