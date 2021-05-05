#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import warnings
import pathlib
import numpy as np
import numpy.ma as ma

from ase.io import read, write
from ase.io.lammpsrun import read_lammps_dump_text
from ase.data import atomic_numbers, atomic_masses

from GDPy.calculator.inputs import LammpsInput
from GDPy.sampler.exploration import sample_configuration

class Sampler():

    """
    Exploration Strategies
        1. random structure sampling
        2. samll pertubations on given structures
        3. molecular dynamics with surrogate potential
        4. molecular dynamics with uncertainty-aware potential
    
    Initial Systems
        initial structures must be manually prepared
    """

    def __init__(self, inputs):
        pass

default_variables = dict(
    nsteps = 0, 
    thermo_freq = 0, 
    dtime = 0.002, # be care with the unit
    temp = 300,
    pres = -1,
    tau_t = 0.1,
    tau_p = 0.5
)

def parse_exploration():

    return

def map_md_variables(exp_dict: dict):
    temperatures = exp_dict.pop('temperatures', None)
    pressures = exp_dict.pop('pressures', None)

    sample_variables = default_variables.copy()
    sample_variables['nsteps'] = exp_dict['nsteps']
    sample_variables['dtime'] = exp_dict['timestep']
    sample_variables['thermo_freq'] = exp_dict.get('freq', 10)
    sample_variables['tau_t'] = exp_dict.get('tau_t', 0.1)
    sample_variables['tau_p'] = exp_dict.get('tau_p', 0.5)

    return temperatures, pressures, sample_variables


def sampler_main(iter_directory: pathlib.Path, main_database, main_dict: dict):
    """"""
    # check dp models
    type_map = main_dict['type_map']
    num_models = main_dict['training']['num_models'] # number of dp models
    ensemble_path = iter_directory / 'ensemble'

    # check all registered explorations
    init_systems = main_dict['systems']
    explorations = main_dict['explorations']

    for exp_name, exp_dict in explorations.items():
        print(exp_name)
        # for test
        if exp_name != 'MD-NVT-Oxides':
            continue
        included_systems = exp_dict.get('systems', None)
        if included_systems is not None:
            # some general parameters
            potential = exp_dict.pop('potential', 'deepmd')
            backend = exp_dict.pop('backend', 'lammps')
            exp_params = exp_dict['params']
            thermostat = exp_params.pop('thermostat', None)
            temperatures, pressures, sample_variables = map_md_variables(exp_params)
            # loop over systems
            for slabel in included_systems:
                system_dict = init_systems[slabel] # system name
                sname = system_dict['name'] # name for the structure data file
                scomp = system_dict['composition'] # system composition
                cons = system_dict.get('constraint', None)
                name_path = iter_directory / exp_name / (slabel+'-'+thermostat)
                # create directories
                if potential == 'deepmd':
                    model = {
                        'deepmd':
                        [(str(ensemble_path)+'/model-%d'+'/graph.pb') %i for i in range(num_models)]
                    }

                    if backend == 'ase':
                        #data = pathlib.Path('/users/40247882/projects/oxides/gdp-main/init-systems') / (comp+'.data')
                        #sample_configuration(data, type_map, model['deepmd'], sample_variables, temperatures)
                        raise NotImplementedError('other backends...')
                    elif backend == 'lammps':
                        # TODO: change the path to a variable
                        data = pathlib.Path('/users/40247882/projects/oxides/gdp-main/init-systems') / (sname+'.data')
                        lmpin = LammpsInput(type_map, data, model, thermostat, sample_variables, cons)
                        if thermostat == 'nvt':
                            for temp in temperatures:
                                temp_dir = name_path / str(temp)
                                try:
                                    temp_dir.mkdir(parents=True)
                                except FileExistsError:
                                    print('skip this %s' %temp_dir)
                                    continue
                                lmpin.temp = temp
                                lmpin.write(temp_dir/'in.lammps')
                        elif thermostat == 'npt':
                            pass
                        else:
                            raise NotImplementedError('no other thermostats')
                    else:
                        raise NotImplementedError('other backends...')
                elif potential == 'reax':
                    # TODO: better use OrderedDict
                    reax_element_map = type_map.copy()
                    for ele, num in scomp.items():
                        if num == 0:
                            reax_element_map.pop(ele, None)
                    for idx, ele in enumerate(reax_element_map.keys()):
                        reax_element_map[ele] = idx
                    model = {
                        'reax/c': main_dict['surrogate']['reax']
                    }
                    if backend == 'ase':
                        raise NotImplementedError('ase does not support reax...')
                    elif backend == 'lammps':
                        data = pathlib.Path('/users/40247882/projects/oxides/gdp-main/init-systems/charge') / (sname+'.data')
                        lmpin = LammpsInput(reax_element_map, data, model, thermostat, sample_variables, cons)
                        if thermostat == 'nvt':
                            for temp in temperatures:
                                temp_dir = name_path / str(temp)
                                try:
                                    temp_dir.mkdir(parents=True)
                                except FileExistsError:
                                    print('skip this %s' %temp_dir)
                                    continue
                                lmpin.temp = temp
                                lmpin.write(temp_dir/'in.lammps')
                        elif thermostat == 'npt':
                            for temp in temperatures:
                                for pres in pressures:
                                    temp_dir = name_path / (str(temp)+'_'+str(pres))
                                    try:
                                        temp_dir.mkdir(parents=True)
                                    except FileExistsError:
                                        print('skip this %s' %temp_dir)
                                        continue
                                    lmpin.temp = temp
                                    lmpin.pres = pres
                                    lmpin.write(temp_dir/'in.lammps')
                        else:
                            raise NotImplementedError('no other thermostats')
                else:
                    raise NotImplementedError('other potentials...')

    exit()

    return

def collect_sample_data(iter_directory: pathlib.Path, main_database, main_dict: dict):
    """collect sample data"""
    # check all registered explorations
    type_map = main_dict['type_map']
    init_systems = main_dict['systems']
    explorations = main_dict['explorations']

    for exp_name, exp_dict in explorations.items():
        # for test
        if exp_name != 'MD-NVT-Oxides':
            continue
        # deviation
        devi = exp_dict.get('deviation', None)
        # backend and potential
        pot = exp_dict['potential']
        md_prefix = iter_directory / exp_name
        if md_prefix.is_dir() and md_prefix.exists():
            included_systems = exp_dict.get('systems', None)
            if included_systems is not None:
                print("checking system %s ..."  %md_prefix)
                exp_params = exp_dict['params']
                thermostat = exp_params.pop('thermostat', None)
                temperatures, pressures, sample_variables = map_md_variables(exp_params)

                # loop over systems
                for slabel in included_systems:
                    # for test
                    #if slabel != 'PtO':
                    #    continue
                    # TODO: better use OrderedDict
                    system_dict = init_systems[slabel] # system name
                    scomp = system_dict['composition'] # system composition
                    elem_map = type_map.copy()
                    for ele, num in scomp.items():
                        if num == 0:
                            elem_map.pop(ele, None)
                    elements = list(elem_map.keys())
                    # check thermostats
                    if thermostat == 'nvt':
                        sys_prefix = md_prefix / (slabel+'-'+thermostat)
                        sorted_path = sys_prefix / 'sorted'
                        if sorted_path.exists():
                            warnings.warn('sorted_path exists', UserWarning)
                        else:
                            sorted_path.mkdir(exist_ok=True)

                        all_frames = []
                        for temp in temperatures:
                            # read dump
                            temp = str(temp)
                            dump_xyz = sys_prefix/temp/'traj.dump'
                            if not dump_xyz.exists():
                                warnings.warn('no %s' %dump_xyz, UserWarning)
                                break
                            frames = read(dump_xyz, ':', 'lammps-dump-text', specorder=elements)[1:]
                            print('nframes at temp %sK: %d' %(temp,len(frames)))
                            # read deviation
                            if devi is not None:
                                low_devi, high_devi = devi
                                devi_out = sys_prefix/temp/'model_devi.out'
                                max_fdevi = np.loadtxt(devi_out)[1:,4]
                                assert len(frames) == max_fdevi.shape[0] # not necessary

                                max_fdevi = max_fdevi.flatten().tolist() # make it a list
                                unlearned_generator = filter(
                                    lambda x: True if low_devi < x[1] < high_devi else False,
                                    zip(frames,max_fdevi)
                                )
                                unlearned_frames = [x[0] for x in list(unlearned_generator)]

                                nlearned = len(list(filter(lambda x: True if x < low_devi else False, max_fdevi)))
                                nfailed = len(list(filter(lambda x: True if x > high_devi else False, max_fdevi)))
                                print(
                                    'learned: %d candidate: %d failed: %d' 
                                    %(nlearned,len(unlearned_frames),nfailed)
                                )
                                # print(unlearned_frames)
                                frames = unlearned_frames
                            else:
                                pass

                            # sometimes all frames have small deviations
                            if frames:
                                out_xyz = str(sorted_path/temp)
                                write(out_xyz+'.xyz', frames)
                                all_frames.extend(frames)

                        print('TOTAL NUMBER OF FRAMES %d in %s' %(len(all_frames),sys_prefix))
                        write(sorted_path/str(slabel+'_ALL.xyz'), all_frames)
                    else:
                        raise NotImplementedError('no other thermostats')

    return

if __name__ == '__main__':
    pass
