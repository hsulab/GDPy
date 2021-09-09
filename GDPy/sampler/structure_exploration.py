#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from abc import ABC
from os import system
from pathlib import Path
import json
import warnings

from typing import Union, Callable

import numpy as np

from ase.io import read, write
from ase.calculators.singlepoint import SinglePointCalculator

from GDPy.selector.structure_selection import calc_feature, cur_selection, select_structures
from GDPy.utils.data import vasp_creator, vasp_collector

class AbstractExplorer(ABC):

    def __init__(self, main_dict: str):
        """"""
        # self.main_dict = main_dict
        self.explorations = main_dict['explorations']
        
        return
    
    def register_calculator(self, pm):
        """ use potentila manager
        """

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
    

class RandomExplorer(AbstractExplorer):

    """
    Quasi-Random Structure Search
        ASE-GA, USPEX, AIRSS
    """

    # collect params
    CONVERGED_FORCE = 0.05
    ENERGY_DIFFERENCE = 3.0 # energy difference compared to the lowest
    ESVAR_TOL = 0.10 # energy standard variance tolerance
    NUM_LOWEST = 200

    # select params


    def __init__(self, pm, main_dict: str,):
        super().__init__(main_dict)

        atypes = None
        self.calc = pm.generate_calculator(atypes)

        return

    def icollect(self, exp_name, working_directory):
        """collect configurations..."""
        exp_dict = self.explorations[exp_name]
        exp_systems = exp_dict["systems"]
        for slabel in exp_systems:
            system_path = working_directory / exp_name / slabel
            print(f"===== {system_path} =====")
            results_path = system_path / "results"
            if not results_path.exists():
                print("results not exists, and skip this system...")
                continue
            selection_path = system_path / "selection"
            if selection_path.exists():
                print("selection exists, and skip this system...")
                continue
            else:
                selection_path.mkdir()
            devi_path = selection_path / "stru_devi.out"

            # start collection and first selection
            candidates = read(
                system_path / "results" / "all_candidates.xyz", ":"
            )
            min_energy = candidates[0].get_potential_energy()
            print("Lowest Energy: ", min_energy)

            with open(devi_path, "w") as fopen:
                fopen.write("# INDEX GAID Energy StandardVariance f_stdvar\n")

            # first select configurations with 
            nconverged = 0
            converged_energies, converged_frames = [], []
            for idx, atoms in enumerate(candidates): 
                # check energy if too large then skip
                confid = atoms.info["confid"]
                # cur_energy = atoms.get_potential_energy()
                # if np.fabs(cur_energy - min_energy) > self.ENERGY_DIFFERENCE:
                #     print("Skip high-energy structure...")
                #     continue
                # GA has no forces for fixed atoms
                forces = atoms.get_forces()
                max_force = np.max(np.fabs(forces))
                if max_force < self.CONVERGED_FORCE:
                    # TODO: check uncertainty
                    self.calc.reset()
                    self.calc.calc_uncertainty = True
                    atoms.calc = self.calc
                    energy = atoms.get_potential_energy()
                    enstdvar = atoms.calc.results["en_stdvar"]
                    if enstdvar < self.ESVAR_TOL:
                        # print(f"{idx} small svar {enstdvar} no need to learn")
                        continue
                    maxfstdvar = np.max(atoms.calc.results["force_stdvar"])
                    # print("Var_En: ", enstdvar)
                    with open(devi_path, "a") as fopen:
                        fopen.write(
                            "{:<8d} {:<8d}  {:>8.4f}  {:>8.4f}  {:>8.4f}\n".format(
                                idx, confid, energy, enstdvar, maxfstdvar
                            )
                        )
                    nconverged += 1
                    # save calc results
                    # results = atoms.calc.results.copy()
                    # new_calc = SinglePointCalculator(atoms, **results)
                    # atoms.calc = new_calc
                    converged_energies.append(energy)
                    converged_frames.append(atoms)
                    # print("converged")
                else:
                    print(f"{idx} found unconverged")
                #if nconverged == self.NUM_LOWEST:
                #    break
            #else:
            #    print("not enough converged structures...")
            
            # boltzmann selection on minima
            # energies = [a.get_potential_energy for a in converged_frames]
            if len(converged_frames) < self.NUM_LOWEST:
                print(
                    "Number of frames is smaller than required. {} < {}".format(
                        len(converged_frames), self.NUM_LOWEST
                    )
                )
                num_sel = int(np.min([self.NUM_LOWEST, 0.5*len(converged_frames)]))
                print(f"adjust number of selected to {num_sel}")
            else:
                num_sel = self.NUM_LOWEST
            selected_frames, selected_props = self.boltzmann_histogram_selection(
                converged_energies, converged_frames, num_sel, 3.0
            )

            # collect trajectories
            elements = ["O", "Pt"]
            traj_frames = []
            for atoms in selected_frames:
                confid = atoms.info["confid"]
                calc_dir = system_path / "tmp_folder" / ("cand"+str(confid))
                dump_file = calc_dir / "surface.dump"
                frames = read(dump_file, ':', 'lammps-dump-text', specorder=elements)
                # print("trajectory length: ", len(frames))
                traj_frames.extend(frames)
            print("TOTAL TRAJ FRAMES: ", len(traj_frames))
            write(selection_path / "all-traj.xyz", traj_frames)

        return
    
    def boltzmann_histogram_selection(self, props, frames, num_minima, kT=-1.0):
        """"""
        # calculate minima properties 
    
        # compute desired probabilities for flattened histogram
        histo = np.histogram(props)
        min_prop = np.min(props)
    
        config_prob = []
        for H in props:
            bin_i = np.searchsorted(histo[1][1:], H)
            if histo[0][bin_i] > 0.0:
                p = 1.0/histo[0][bin_i]
            else:
                p = 0.0
            if kT > 0.0:
                p *= np.exp(-(H-min_prop)/kT)
            config_prob.append(p)
        
        assert len(config_prob) == len(props)
    
        selected_frames = []
        for i in range(num_minima):
            # TODO: rewrite by mask 
            config_prob = np.array(config_prob)
            config_prob /= np.sum(config_prob)
            cumul_prob = np.cumsum(config_prob)
            rv = np.random.uniform()
            config_i = np.searchsorted(cumul_prob, rv)
            #print(converged_trajectories[config_i][0])
            selected_frames.append(frames[config_i])
    
            # remove from config_prob by converting to list
            config_prob = list(config_prob)
            del config_prob[config_i]
    
            # remove from other lists
            del props[config_i]
            del frames[config_i]
            
        return selected_frames, props
    
    def iselect(self, exp_name, working_directory):
        """select data from single calculation"""
        exp_dict = self.explorations[exp_name]

        included_systems = exp_dict.get("systems", None)
        if included_systems is not None:
            exp_path = working_directory / exp_name

            selected_numbers = exp_dict["selection"]["num"]
            if isinstance(selected_numbers, list):
                assert len(selected_numbers) == len(included_systems), "each system must have a number"
            else:
                selected_numbers = selected_numbers * len(included_systems)

            # loop over systems
            for slabel, num in zip(included_systems, selected_numbers):
                # check valid
                if num <= 0:
                    print("selected number is zero...")
                    continue
                # select
                sys_prefix = exp_path / slabel
                print("checking system %s ..."  %sys_prefix)
                selected_frames = self.perform_cur(sys_prefix, slabel, exp_dict, num)
                if selected_frames is None:
                    print("No candidates in {0}".format(sys_prefix))
                else:
                    write(sys_prefix / (slabel + '-tot-sel.xyz'), selected_frames)

        return
    
    def perform_cur(self, cur_prefix, slabel, exp_dict, num):
        """"""
        soap_parameters = exp_dict['selection']['soap']
        njobs = exp_dict['selection']['njobs']
        zeta, strategy = exp_dict['selection']['selection']['zeta'], exp_dict['selection']['selection']['strategy']

        # assert soap_parameters["species"] == self.type_list

        sorted_path = cur_prefix / "selection"
        print("===== selecting system %s =====" %cur_prefix)
        if sorted_path.exists():
            all_xyz = sorted_path / "all-traj.xyz"
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
            selected_numbers = exp_dict["selection"]["num"]
            if isinstance(selected_numbers, list):
                assert len(selected_numbers) == len(included_systems), "each system must have a number"
            else:
                selected_numbers = selected_numbers * len(included_systems)

            for slabel, num in zip(included_systems, selected_numbers):
                if num <= 0:
                    print("selected number is zero...")
                    continue

                name_path = working_directory / exp_name / (slabel) # system directory
                # create all calculation dirs
                sorted_path = name_path / "selection" # directory with collected xyz configurations
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
                        Path(prefix),
                        "/users/40247882/repository/GDPy/GDPy/utils/data/vasp_calculator.py",
                        incar_template,
                        collected_path
                    )
                else:
                    warnings.warn("There is no %s." %collected_path, UserWarning)

        return


def run_exploration(pot_json, exp_json, chosen_step, global_params = None):
    from GDPy.potential.manager import create_manager
    pm = create_manager(pot_json)
    print(pm.models)

    # create exploration
    with open(exp_json, 'r') as fopen:
        exp_dict = json.load(fopen)
    
    scout = RandomExplorer(pm, exp_dict)

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
    

if __name__ == "__main__":
    # test
    pot_json = "/mnt/scratch2/users/40247882/oxides/eann-main/reduce-12/validations/potential.json"
    exp_json = "/mnt/scratch2/users/40247882/oxides/eann-main/exp-ga22.json"
    #chosen_step = "collect"
    #chosen_step = "select"
    chosen_step = "calc"
    run_exploration(pot_json, exp_json, chosen_step)