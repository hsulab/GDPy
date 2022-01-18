#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import time
import shutil
import json
import sys
from typing import Counter, Union
import warnings
import pathlib
from joblib import Parallel, delayed
import numpy as np
import numpy.ma as ma

from ase import Atoms
from ase.io import read, write
from ase.io.lammpsrun import read_lammps_dump_text
from ase.data import atomic_numbers, atomic_masses
from numpy.ma.extras import isin

from GDPy.calculator.ase_interface import AseInput
from GDPy.calculator.inputs import LammpsInput
from GDPy.selector.structure_selection import calc_feature, cur_selection, select_structures

from GDPy.machine.machine import SlurmMachine

from GDPy.utils.data import vasp_creator, vasp_collector


class Sampler():

    """
    Exploration Strategies
        1. random structure sampling
        2. samll pertubations on given structures
        3. molecular dynamics with surrogate potential
        4. molecular dynamics with uncertainty-aware potential
    
    Initial Systems
        initial structures must be manually prepared
    
    Units
        fs, eV, eV/AA
    """

    supported_potentials = ["reax", "deepmd", "eann"]
    supported_procedures = ["create", "collect", "select", "calculate"]

    # set default variables
    # be care with the unit
    default_variables = dict(
        nsteps = 0, 
        thermo_freq = 0, 
        dtime = 0.002, # ps
        temp = 300, # Kelvin
        pres = -1, # bar
        tau_t = 0.1, # ps
        tau_p = 0.5 # ps
    )

    default_params = {
        "collect": {
            "deviation": None
        }
    }

    def __init__(self, pm, main_dict: dict):
        """"""
        self.pot_manager = pm
        self.type_map = main_dict['type_map']
        self.type_list = list(self.type_map.keys())
        self.explorations = main_dict['explorations']
        self.init_systems = main_dict['systems']

        assert self.pot_manager.type_map == self.type_map, 'type map should be consistent'

        # for job prefix
        self.job_prefix = ""

        return
    
    @staticmethod
    def map_md_variables(default_variables, exp_dict: dict, unit='default'):
        
        # update variables
        temperatures = exp_dict.pop('temperatures', None)
        pressures = exp_dict.pop('pressures', None)

        sample_variables = default_variables.copy()
        sample_variables['nsteps'] = exp_dict['nsteps']
        sample_variables['dtime'] = exp_dict['timestep']
        sample_variables['thermo_freq'] = exp_dict.get('freq', 10)
        sample_variables['tau_t'] = exp_dict.get('tau_t', 0.1)
        sample_variables['tau_p'] = exp_dict.get('tau_p', 0.5)

        return temperatures, pressures, sample_variables
    
    def run(self, operator, working_directory): 
        """create for all explorations"""
        working_directory = pathlib.Path(working_directory)
        self.job_prefix = working_directory.resolve().name # use resolve to get abspath
        print("job prefix: ", self.job_prefix)
        for exp_name in self.explorations.keys():
            exp_directory = working_directory / exp_name
            # note: check dir existence in sub function
            operator(exp_name, working_directory)

        return 

    def icreate(self, exp_name, working_directory):
        """create for each exploration"""
        exp_dict = self.explorations[exp_name]
        job_script = exp_dict.get('jobscript', None)
        included_systems = exp_dict.get('systems', None)
        if included_systems is not None:
            # check potential parameters
            #potential = exp_dict.pop("potential", None) 
            #if potential not in self.supported_potentials:
            #    raise ValueError("Potential %s is not supported..." %potential)

            # MD exploration params
            exp_params = exp_dict['params']
            thermostat = exp_params.pop('thermostat', None)
            temperatures, pressures, sample_variables = self.map_md_variables(self.default_variables, exp_params) # be careful with units
            # loop over systems
            for slabel in included_systems:
                system_dict = self.init_systems[slabel] # system name
                structure = system_dict['structure']
                scomp = system_dict['composition'] # system composition
                atypes = []
                for atype, number in scomp.items():
                    if number > 0:
                        atypes.append(atype)

                cons = system_dict.get('constraint', None)
                name_path = working_directory / exp_name / (slabel+'-'+thermostat)
                # create directories
                # check single data or a list of structures
                runovers = [] # [(structure,working_dir),...,()]
                if structure.endswith('.data'):
                    runovers.append((structure,name_path))
                else:
                    data_path = pathlib.Path(system_dict['structure'])
                    for f in data_path.glob(slabel+'*'+'.data'):
                        cur_path = name_path / f.stem
                        runovers.append((f, cur_path))
                # create all 
                calc_input = self.create_input(self.pot_manager, atypes, sample_variables) # use inputs with preset md params
                for (stru_path, work_path) in runovers:
                    self.create_exploration(
                        work_path, job_script, calc_input, stru_path, cons, temperatures, pressures
                    )

        return
    
    def create_input(
        self, pot_manager, 
        atypes: list,
        md_params: dict
    ):
        """ create calculation input object
        """
        # create calculation input object
        calc = pot_manager.generate_calculator(atypes)
        if pot_manager.backend == "ase":
            calc_input = AseInput(atypes, calc, md_params)
        elif pot_manager.backend == "lammps":
            calc_input = LammpsInput(atypes, calc, md_params)

        return calc_input
    
    def create_exploration(self, name_path, job_script, calc_input, structure, cons, temperatures, pressures):
        """"""
        try:
            name_path.mkdir(parents=True)
            print('create this %s' %name_path)
            # create job script
            if job_script is not None:
                job_script = pathlib.Path(job_script)
                ## shutil.copy(job_script, name_path / job_script.name)
                #with open(name_path / job_script.name, 'w') as fopen:
                #    fopen.write(create_test_slurm(name_path.name))
                slurm = SlurmMachine(job_script)
                slurm.machine_dict["job-name"] = self.job_prefix + "-" + name_path.name
                slurm.write(name_path / job_script.name)

        except FileExistsError:
            print('skip this %s' %name_path)
            return
        
        # bind structure
        calc_input.bind_structure(structure, cons)
        
        # create input directories with various thermostats
        if calc_input.thermostat == 'nvt':
            for temp in temperatures:
                temp_dir = name_path / str(temp)
                try:
                    temp_dir.mkdir(parents=True)
                except FileExistsError:
                    print('skip this %s' %temp_dir)
                    continue
                calc_input.temp = temp
                calc_input.write(temp_dir)
        elif calc_input.thermostat == 'npt':
            for temp in temperatures:
                for pres in pressures:
                    temp_dir = name_path / (str(temp)+'_'+str(pres))
                    try:
                        temp_dir.mkdir(parents=True)
                    except FileExistsError:
                        print('skip this %s' %temp_dir)
                        continue
                    calc_input.temp = temp
                    calc_input.pres = pres
                    calc_input.write(temp_dir)
        else:
            raise NotImplementedError('no other thermostats')
        
        # TODO: submit job
        # if not dry-run
        output = slurm.submit(name_path / job_script.name)
        print("try to submit: ", name_path / job_script.name)
        print(output)

        return
    
    def icollect(self, exp_name, working_directory, skipped_systems=[]):
        """collect data from single calculation"""
        exp_dict = self.explorations[exp_name]
        # deviation
        if self.default_params["collect"]["deviation"] is None:
            devi = exp_dict.get('deviation', None)
        else:
            devi = self.default_params["collect"]["deviation"]
            print("deviation: ", devi)

        included_systems = exp_dict.get('systems', None)
        if included_systems is not None:
            md_prefix = working_directory / exp_name
            print("checking system %s ..."  %md_prefix)
            exp_params = exp_dict['params']
            thermostat = exp_params.pop('thermostat', None)
            temperatures, pressures, sample_variables = self.map_md_variables(self.default_variables, exp_params) # be careful with units

            # loop over systems
            for slabel in included_systems:
                # TODO: make this into system
                if slabel in skipped_systems:
                    continue
                # TODO: better use OrderedDict
                system_dict = self.init_systems[slabel] # system name
                scomp = system_dict['composition'] # system composition
                elem_map = self.type_map.copy()
                for ele, num in scomp.items():
                    if num == 0:
                        elem_map.pop(ele, None)
                elements = list(elem_map.keys())
                # check thermostats
                if thermostat == 'nvt':
                    sys_prefix = md_prefix / (slabel+'-'+thermostat)
                    
                    if system_dict.get('structures', None):
                        # run over many structures
                        data_path = pathlib.Path(system_dict['structures'][0])
                        nconfigs = len(list(data_path.glob(slabel+'*'+'.data'))) # number of starting configurations
                        for i in range(nconfigs):
                            cur_prefix = sys_prefix / (slabel + '-' + str(i))
                            # make sort dir
                            sorted_path = cur_prefix / 'sorted'
                            print("===== collecting system %s =====" %cur_prefix)
                            if sorted_path.exists():
                                self.override = True
                                if self.override:
                                    warnings.warn('sorted_path removed in %s' %cur_prefix, UserWarning)
                                    shutil.rmtree(sorted_path)
                                    sorted_path.mkdir()
                                else:
                                    warnings.warn('sorted_path exists in %s' %cur_prefix, UserWarning)
                                    continue
                            else:
                                sorted_path.mkdir()
                            # extract frames
                            all_frames = []
                            for temp in temperatures:
                                # read dump
                                temp = str(temp)
                                dump_xyz = cur_prefix/temp/'traj.dump'
                                if dump_xyz.exists():
                                    frames = read(dump_xyz, ':', 'lammps-dump-text', specorder=elements)[1:]
                                else:
                                    dump_xyz = cur_prefix/temp/'traj.xyz'
                                    if dump_xyz.exists():
                                        frames = read(dump_xyz, ':')[1:]
                                    else:
                                        warnings.warn('no trajectory file in %s' %dump_xyz, UserWarning)
                                        continue
                                print('nframes at temp %sK: %d' %(temp,len(frames)))

                                frames = self.extract_deviation(cur_prefix/temp, frames, devi)

                                # sometimes all frames have small deviations
                                if frames:
                                    out_xyz = str(sorted_path/temp)
                                    write(out_xyz+'.xyz', frames)
                                    all_frames.extend(frames)

                            print('TOTAL NUMBER OF FRAMES %d in %s' %(len(all_frames),cur_prefix))
                            write(sorted_path/str(slabel+'_ALL.xyz'), all_frames)
                    else:
                        # make sort dir
                        sorted_path = sys_prefix / "sorted"
                        print("===== collecting system %s =====" %sys_prefix)
                        if sorted_path.exists():
                            override = True
                            if override:
                                warnings.warn('sorted_path removed in %s' %sys_prefix, UserWarning)
                                shutil.rmtree(sorted_path)
                                sorted_path.mkdir()
                            else:
                                warnings.warn('sorted_path exists in %s' %sys_prefix, UserWarning)
                                continue
                        else:
                            sorted_path.mkdir()
                        # extract frames
                        all_frames = []
                        for temp in temperatures:
                            # read dump
                            temp = str(temp)
                            dump_xyz = sys_prefix/temp/'traj.dump'
                            if dump_xyz.exists():
                                frames = read(dump_xyz, ':', 'lammps-dump-text', specorder=elements)[1:]
                            else:
                                dump_xyz = sys_prefix/temp/'traj.xyz'
                                if dump_xyz.exists():
                                    frames = read(dump_xyz, ':')[1:]
                                else:
                                    warnings.warn('no trajectory file in %s' %dump_xyz, UserWarning)
                                    continue
                            print('nframes at temp %sK: %d' %(temp,len(frames)))

                            frames = self.extract_deviation(sys_prefix/temp, frames, devi)

                            # sometimes all frames have small deviations
                            if frames:
                                out_xyz = str(sorted_path/temp)
                                write(out_xyz+'.xyz', frames)
                                all_frames.extend(frames)

                        print('TOTAL NUMBER OF FRAMES %d in %s' %(len(all_frames),sys_prefix))
                        if len(all_frames) > 0:
                            write(sorted_path/str(slabel+'_ALL.xyz'), all_frames)
                else:
                    raise NotImplementedError('no other thermostats')

        return
    
    def extract_deviation(self, cur_dir, frames, devi=None):
        # read deviation
        if devi is not None:
            low_devi, high_devi = devi
            devi_out = cur_dir / 'model_devi.out'
            # TODO: DP and EANN has different formats
            # max_fdevi = np.loadtxt(devi_out)[1:,4] # DP
            max_fdevi = np.loadtxt(devi_out)[1:,5] # EANN

            err =  '%d != %d' %(len(frames), max_fdevi.shape[0])
            assert len(frames) == max_fdevi.shape[0], err # not necessary

            max_fdevi = max_fdevi.flatten().tolist() # make it a list
            unlearned_generator = filter(
                lambda x: True if low_devi < x[1] < high_devi else False,
                zip(frames,max_fdevi)
            )
            unlearned_frames = [x[0] for x in list(unlearned_generator)]

            nlearned = len(list(filter(lambda x: True if x < low_devi else False, max_fdevi)))
            nfailed = len(list(filter(lambda x: True if x > high_devi else False, max_fdevi)))
            print(
                'learned: %d candidate: %d failed: %d\n' 
                %(nlearned,len(unlearned_frames),nfailed)
            )
            # print(unlearned_frames)
            frames = unlearned_frames
        else:
            pass

        return frames
    
    def iselect(self, exp_name, working_directory):
        """select data from single calculation"""
        exp_dict = self.explorations[exp_name]

        #pattern = "surf-9O*"
        pattern = "O*"

        included_systems = exp_dict.get('systems', None)
        if included_systems is not None:
            md_prefix = working_directory / exp_name
            print("checking system %s ..."  %md_prefix)
            exp_params = exp_dict['params']
            thermostat = exp_params.pop('thermostat', None)
            #temperatures, pressures, sample_variables = self.map_md_variables(self.default_variables, exp_params) # be careful with units

            selected_numbers = exp_dict["selection"]["num"]
            if isinstance(selected_numbers, list):
                assert len(selected_numbers) == len(included_systems), "each system must have a number"
            else:
                selected_numbers = selected_numbers * len(included_systems)

            # loop over systems
            for slabel, num in zip(included_systems, selected_numbers):
                if num <= 0:
                    print("selected number is zero...")
                    continue
                if re.match(pattern, slabel):
                    # TODO: better use OrderedDict
                    system_dict = self.init_systems[slabel] # system name
                    if thermostat == "nvt":
                        sys_prefix = md_prefix / (slabel+'-'+thermostat)
                        if False: # run over configurations
                            sorted_dirs = []
                            for p in sys_prefix.glob(pattern):
                                sorted_dirs.append(p)
                            sorted_dirs.sort()

                            total_selected_frames = []
                            for p in sorted_dirs:
                                print(p)
                                selected_frames = self.perform_cur(p, slabel, exp_dict)
                                total_selected_frames.extend(selected_frames)
                            write(sys_prefix / (slabel + '-tot-sel.xyz'), total_selected_frames)

                        else:
                            selected_frames = self.perform_cur(sys_prefix, slabel, exp_dict, num)
                            if selected_frames is None:
                                print("No candidates in {0}".format(sys_prefix))
                            else:
                                write(sys_prefix / (slabel + '-tot-sel.xyz'), selected_frames)
                    else:
                        # TODO: npt
                        pass
                else:
                    warnings.warn('%s is not valid for the pattern %s.' %(slabel, pattern), UserWarning)

        return
    
    def perform_cur(self, cur_prefix, slabel, exp_dict, num):
        """"""
        soap_parameters = exp_dict['selection']['soap']
        njobs = exp_dict['selection']['njobs']
        zeta, strategy = exp_dict['selection']['selection']['zeta'], exp_dict['selection']['selection']['strategy']

        sorted_path = cur_prefix / 'sorted'
        print("===== selecting system %s =====" %cur_prefix)
        if sorted_path.exists():
            all_xyz = sorted_path / str(slabel+'_ALL.xyz')
            if all_xyz.exists():
                print('wang')
                # read structures and calculate features 
                frames = read(all_xyz, ':')
                features_path = sorted_path / 'features.npy'
                print(features_path.exists())
                if features_path.exists():
                    features = np.load(features_path)
                    assert features.shape[0] == len(frames)
                else:
                    print('start calculating features...')
                    features = calc_feature(frames, soap_parameters, njobs, features_path)
                    print('finished calculating features...')
                # cur decomposition 
                cur_scores, selected = cur_selection(features, num, zeta, strategy)
                content = '# idx cur sel\n'
                for idx, cur_score in enumerate(cur_scores):
                    stat = 'F'
                    if idx in selected:
                        stat = 'T'
                    content += '{:>12d}  {:>12.8f}  {:>2s}\n'.format(idx, cur_score, stat) 
                with open(sorted_path / 'cur_scores.txt', 'w') as writer:
                    writer.write(content)

                selected_frames = []
                print("Writing structure file... ")
                for idx, sidx in enumerate(selected):
                    selected_frames.append(frames[int(sidx)])
                write(sorted_path / (slabel+'-sel.xyz'), selected_frames)
                print('')
            else:
                # no candidates
                selected_frames = None
        else:
            raise ValueError('miaow')
        
        return selected_frames
    
    def icalc(self, exp_name, working_directory):
        """calculate configurations with reference method"""
        exp_dict = self.explorations[exp_name]

        # some parameters
        calc_dict = exp_dict["calculation"]
        nstructures = calc_dict.get("nstructures", 100000) # number of structures in each calculation dirs
        incar_template = calc_dict.get("incar")

        prefix = working_directory / (exp_name + "-fp")
        if prefix.exists():
            warnings.warn("fp directory exists...", UserWarning)
        else:
            prefix.mkdir(parents=True)

        # start 
        included_systems = exp_dict.get('systems', None)
        if included_systems is not None:
            # MD exploration params
            exp_params = exp_dict['params']
            thermostat = exp_params.pop("thermostat", None)

            selected_numbers = exp_dict["selection"]["num"]
            if isinstance(selected_numbers, list):
                assert len(selected_numbers) == len(included_systems), "each system must have a number"
            else:
                selected_numbers = selected_numbers * len(included_systems)

            for slabel, num in zip(included_systems, selected_numbers):
                if num <= 0:
                    print("selected number is zero...")
                    continue
                system_dict = self.init_systems[slabel] # system name
                structure = system_dict["structure"]
                scomp = system_dict["composition"] # system composition
                atypes = []
                for atype, number in scomp.items():
                    if number > 0:
                        atypes.append(atype)

                name_path = working_directory / exp_name / (slabel+'-'+thermostat) # system directory
                # create directories
                # check single data or a list of structures
                runovers = [] # [(structure,working_dir),...,()]
                if structure.endswith('.data'):
                    runovers.append((structure,name_path))
                else:
                    data_path = pathlib.Path(system_dict['structure'])
                    for f in data_path.glob(slabel+'*'+'.data'):
                        cur_path = name_path / f.stem
                        runovers.append((f, cur_path))
                # create all calculation dirs
                for (stru_path, name_path) in runovers:
                    sorted_path = name_path / "sorted" # directory with collected xyz configurations
                    collected_path = sorted_path / (slabel + "-sel.xyz")
                    if collected_path.exists():
                        print("use selected frames...")
                    else:
                        print("use all candidates...")
                        collected_path = sorted_path / (slabel + "_ALL.xyz")
                    if collected_path.exists():
                        #frames = read(collected_path, ":")
                        #print("There are %d configurations in %s." %(len(frames), collected_path))
                        vasp_creator.create_files(
                            pathlib.Path(prefix),
                            "/users/40247882/repository/GDPy/GDPy/utils/data/vasp_calculator.py",
                            incar_template,
                            collected_path
                        )
                    else:
                        warnings.warn("There is no %s." %collected_path, UserWarning)

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
        main_database = pathlib.Path("/users/40247882/scratch2/PtOx-dataset")

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
                # check system
                atoms = frames[0]
                c = Counter(atoms.get_chemical_symbols())
                #print(c)
                sys_name_list = []
                for s in self.type_list:
                    sys_name_list.append(s)
                    num = c.get(s, 0)
                    sys_name_list.append(str(num))
                sys_name = "".join(sys_name_list)
                #print(sys_name)
                out_name = main_database / sys_name / (d.name + "-" + pot_gen + ".xyz")
                write(out_name, frames)
            else:
                print("No frames...")

        return
    

def run_exploration(pot_json, exp_json, chosen_step, global_params = None):
    from GDPy.potential.manager import create_manager
    pm = create_manager(pot_json)
    print(pm.models)

    # create exploration
    with open(exp_json, 'r') as fopen:
        exp_dict = json.load(fopen)
    
    method = exp_dict.get("method", "MD")
    if method == "MD":
        scout = Sampler(pm, exp_dict)
    elif method == "GA":
        from GDPy.exploration.structure_exploration import RandomExplorer
        scout = RandomExplorer(pm, exp_dict)
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
