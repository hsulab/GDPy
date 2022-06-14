#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
from pathlib import Path
from collections import Counter
from typing import List

import shutil
import warnings

import numpy as np

from joblib import Parallel, delayed

import ase
from ase import Atoms
from ase import build
from ase.io import read, write
from ase.constraints import FixAtoms
from ase.calculators.singlepoint import SinglePointCalculator

from GDPy.potential.manager import PotManager
from GDPy.utils.command import parse_input_file
from GDPy.graph.creator import StruGraphCreator, SiteGraphCreator
from GDPy.graph.utils import unique_chem_envs, compare_chem_envs
from GDPy.graph.graph_main import create_structure_graphs, add_adsorbate, del_adsorbate, exchange_adsorbate
from GDPy.graph.utils import unpack_node_name
from GDPy.graph.para import paragroup_unique_chem_envs

from ase.ga.standard_comparators import EnergyComparator
from ase.ga.standard_comparators import InteratomicDistanceComparator, get_sorted_dist_list

from .abstract import AbstractExplorer
from GDPy import config
from GDPy.calculator.dynamics import AbstractDynamics
from GDPy.selector.abstract import create_selector


class StructureGenerator():

    def __init__(self, *args, **kwargs):

        return

class AdsorbateGraphGenerator(StructureGenerator):

    """ generate initial configurations with 
        different adsorbates basedo on graphs
    """


    def __init__(self, comp_dict: dict, graph_dict: dict, directory=Path.cwd()):
        """"""
        # --- unpack some params
        # TODO: only support one species now
        for data in comp_dict:
            self.species = data["species"] # atom or molecule
            self.action = data["action"]
            self.distance_to_site = data.get("distance_to_site", 1.5)
            break
        else:
            pass

        self.check_site_unique = graph_dict.pop("check_site_unique", True)
        self.graph_params = graph_dict

        self.directory = Path(directory)

        self.njobs = config.NJOBS

        return
    
    def run(self, frames) -> List[Atoms]:
        """"""
        if self.action == "add":
            created_frames = self.add_adsorbate(frames, self.species, self.distance_to_site)
        elif self.action == "delete":
            # TODO: fix bug
            created_frames = self.del_adsorbate(frames)
        elif self.action == "exchange":
            # TODO: fix bug
            print("---run exchange---")
            selected_indices = self.mut_params["selected_indices"]
            print("for atom indices ", selected_indices)
            ads_species, target_species = self.mut_content.split("->")
            ads_species = ads_species.strip()
            assert ads_species == self.ads_chem_sym, "adsorbate species is not consistent"
            target_species = target_species.strip()
            created_frames = self.exchange_adsorbate(
                frames, target_species, selected_indices=selected_indices
            )

        print(f"number of adsorbate structures: {len(created_frames)}")
        # add confid
        for i, a in enumerate(created_frames):
            a.info["confid"] = i
        ncandidates = len(created_frames)
        
        # - compare frames based on graphs
        selected_indices = [None]*len(created_frames)
        if self.action == "exchange":
            # find target species
            for fidx, a in enumerate(created_frames):
                s = []
                for i, x in enumerate(a.get_chemical_symbols()):
                    if x == target_species:
                        s.append(i)
                selected_indices[fidx] = s

        unique_groups = self.compare_graphs(created_frames, selected_indices=selected_indices)
        print(f"number of unique groups: {len(unique_groups)}")

        # -- unique info
        unique_data = []
        for i, x in enumerate(unique_groups):
            data = ["ug"+str(i)]
            data.extend([a[0] for a in x])
            unique_data.append(data)
        content = "# unique, indices\n"
        content += f"# ncandidates {ncandidates}\n"
        for d in unique_data:
            content += ("{:<8s}  "+"{:<8d}  "*(len(d)-1)+"\n").format(*d)

        unique_info_path = self.directory / "unique-g.txt"
        with open(unique_info_path, "w") as fopen:
            fopen.write(content)

        # --- distance

        # only calc unique ones
        unique_candidates = [] # graphly unique
        for x in unique_groups:
            unique_candidates.append(x[0][1])

        #return created_frames
        return unique_candidates

    def add_adsorbate(self, frames, species: str, distance_to_site: float = 1.5):
        """"""
        print("start adsorbate addition")
        # - build adsorbate
        adsorbate = None
        if species in ase.data.chemical_symbols:
            adsorbate = Atoms(species, positions=[[0.,0.,0.]])
        elif species in ase.collections.g2.names:
            adsorbate = build.molecule(species)
        else:
            raise ValueError(f"Cant create species {species}")

        # joblib version
        st = time.time()

        ads_frames = Parallel(n_jobs=self.njobs)(
            delayed(add_adsorbate)(
                self.graph_params, idx, a, adsorbate, distance_to_site, check_unique=self.check_site_unique
            ) for idx, a in enumerate(frames)
        )
        #print(ads_frames)

        created_frames = []
        for af in ads_frames:
            created_frames.extend(af)

        et = time.time()
        print("add_adsorbate time: ", et - st)

        return created_frames
    
    def del_adsorbate(self, frames):
        """ delete valid adsorbates and
            check graph differences
        """
        # joblib version
        st = time.time()

        ads_frames = Parallel(n_jobs=self.njobs)(delayed(del_adsorbate)(self.graph_params, a, self.ads_chem_sym) for idx, a in enumerate(frames))
        #print(ads_frames)

        created_frames = []
        for af in ads_frames:
            created_frames.extend(af)

        et = time.time()
        print("del_adsorbate time: ", et - st)

        return created_frames
    
    def exchange_adsorbate(self, frames, target_species, selected_indices=None):
        """ change an adsorbate to another species
        """
        # joblib version
        st = time.time()

        ads_frames = Parallel(n_jobs=self.njobs)(
            delayed(exchange_adsorbate)(self.graph_params, a, self.ads_chem_sym, target_species, selected_indices) for a in frames
        )
        #print(ads_frames)

        created_frames = []
        for af in ads_frames:
            created_frames.extend(af)

        et = time.time()
        print("exg_adsorbate time: ", et - st)

        return created_frames

    def compare_graphs(self, frames, graph_params=None, selected_indices=None):
        """"""
        # TODO: change this into a selector
        # calculate chem envs
        st = time.time()

        if graph_params is None:
            graph_params = self.graph_params

        chem_groups = Parallel(n_jobs=self.njobs)(
            delayed(create_structure_graphs)(graph_params, idx, a, s) for idx, (a, s) in enumerate(zip(frames,selected_indices))
        )

        et = time.time()
        print("calc chem envs: ", et - st)
    
        # compare chem envs
        #unique_envs, unique_groups = unique_chem_envs(
        #    chem_groups, list(enumerate(frames))
        #)
        unique_envs, unique_groups = paragroup_unique_chem_envs(
            chem_groups, list(enumerate(frames)), directory=self.directory, n_jobs=self.njobs
        )

        print("number of unique groups: ", len(unique_groups))

        et = time.time()
        print("cmp chem envs: ", et - st)

        return unique_groups


class AdsorbateEvolution(AbstractExplorer):

    action_order = ["adsorption", "dynamics"]

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
    
    def _parse_specific_params(self, input_dict: dict):
        """"""
        self.graph_params = input_dict["system"]["graph"]

        # - parse substrate 
        #self.substrates = read(input_dict["system"]["substrate"], ":")
        self.substrates = read(input_dict["system"]["substrate"]["file"], ":")
        if isinstance(self.substrates, Atoms): # NOTE: if one structure
            self.substrates = [self.substrates]
        print("number of substrates: ", len(self.substrates))

        # create adsorbate
        self.__parse_composition(input_dict["system"]["composition"])

        # selection
        self.energy_cutoff = input_dict["selection"]["energy_cutoff"]

        return
    
    def __parse_composition(self, compos_list: list):
        """ parse composition that operates on the systems
        """
        # TODO: move this to add mutation
        #composition = input_dict["system"]["composition"]
        #assert len(composition) == 1, "only one element is support"
        #self.ads_chem_sym = list(composition.keys())[0]
        #self.adsorbate = Atoms(self.ads_chem_sym, positions=[[0., 0., 0.]])
        #self.ads_number = composition[self.ads_chem_sym]

        self.compositions = compos_list
        print("number of compositions: ", len(self.compositions))
        assert len(self.compositions) == 1, "only support one adsorbate for now."

        return
    
    def _parse_action(self, action_param_list: list, directory) -> dict:
        """"""
        action_dict = {}
        for act in action_param_list:
            assert len(act) == 1, "Each action should only have one param dict."
            act_name, act_params = list(act.items())[0]
            #print(act_Iname, act_params)
            if act_name == "adsorption":
                action = AdsorbateGraphGenerator(act_params["composition"], act_params["graph"], directory)
            elif act_name == "dynamics":
                action = self.pot_manager.create_worker(act_params)
            else:
                pass
            action_dict[act_name] = action

        return action_dict

    def icreate(self, exp_name, working_directory):
        """ perform a list of actions on input system
        """
        # - a few info
        exp_dict = self.explorations[exp_name]
        included_systems = exp_dict.get("systems", None)

        if included_systems is not None:
            for slabel in included_systems:
                # - prepare output directory
                res_dir = working_directory / exp_name / slabel
                if not res_dir.exists():
                    res_dir.mkdir(parents=True)
                else:
                    pass

                # - parse actions
                actions = self._parse_action(exp_dict["action"], res_dir)
                    
                # - read substrate
                system_dict = self.init_systems.get(slabel, None) # system name
                if system_dict is None:
                    raise ValueError(f"Find unexpected system {system_dict}.")
                sys_cons_text = system_dict.get("constraint", None)

                # - read structures
                # the expedition can start with different initial configurations
                stru_path = system_dict["structure"]
                frames = read(stru_path, ":")
                
                # - act
                cur_frames = frames
                for act_name in self.action_order:
                    action = actions.get(act_name, None)
                    act_outpath = res_dir / f"graph-act-{act_name}.xyz"
                    if action is None:
                        continue
                    if isinstance(action, StructureGenerator):
                        if not act_outpath.exists():
                            cur_frames = action.run(cur_frames)
                        else:
                            cur_frames = read(act_outpath, ":")
                    elif isinstance(action, AbstractDynamics):
                        tmp_folder = res_dir / "tmp_folder"
                        if not tmp_folder.exists():
                            tmp_folder.mkdir()
                        new_frames = []
                        for i, atoms in enumerate(cur_frames): 
                            confid = atoms.info["confid"]
                            action.set_output_path(tmp_folder/("cand"+str(confid)))
                            # TODO: check existed results before running
                            new_atoms = action.run(atoms, extra_info=dict(confid=i), constraint=sys_cons_text)
                            new_frames.append(new_atoms)
                        cur_frames = new_frames
                    else:
                        pass
                    write(act_outpath, cur_frames)

        return

    def icollect(self, exp_name, working_directory):
        """ perform a list of actions on input system
        """
        # - a few info
        exp_dict = self.explorations[exp_name]
        included_systems = exp_dict.get("systems", None)

        collection_params = exp_dict["collection"]
        traj_period = collection_params.get("traj_period", 1)
        print(f"traj_period: {traj_period}")

        if included_systems is not None:
            for slabel in included_systems:
                # - prepare output directory
                res_dir = working_directory / exp_name / slabel
                if not res_dir.exists():
                    res_dir.mkdir(parents=True)
                else:
                    pass

                # - parse actions
                actions = self._parse_action(exp_dict["action"], directory=res_dir)
                    
                # - read substrate
                system_dict = self.init_systems.get(slabel, None) # system name
                if system_dict is None:
                    raise ValueError(f"Find unexpected system {system_dict}.")
                sys_cons_text = system_dict.get("constraint", None)

                # - read structures
                # the expedition can start with different initial configurations
                stru_path = system_dict["structure"]
                frames = read(stru_path, ":")
                
                # - act, retrieve trajectory frames
                # TODO: more general interface not limited to dynamics
                traj_frames_path = res_dir / "traj_frames.xyz"
                traj_indices_path = res_dir / "traj_indices.npy"
                if not traj_frames_path.exists():
                    traj_indices = [] # use traj indices to mark selected traj frames
                    all_traj_frames = []
                    tmp_folder = res_dir / "tmp_folder"
                    action = actions["dynamics"]
                    optimised_frames = read(res_dir/"graph-act-dynamics.xyz", ":")
                    # TODO: change this to joblib
                    for atoms in optimised_frames:
                        confid = atoms.info["confid"]
                        action.set_output_path(tmp_folder/("cand"+str(confid)))
                        traj_frames = action._read_trajectory(atoms, label_steps=True)
                        # --- generate indices
                        cur_nframes = len(all_traj_frames)
                        cur_indices = list(range(0,len(traj_frames)-1,traj_period)) + [len(traj_frames)-1]
                        cur_indices = [c+cur_nframes for c in cur_indices]
                        traj_indices.extend(cur_indices)
                        # --- add frames
                        all_traj_frames.extend(traj_frames)
                    np.save(traj_indices_path, traj_indices)
                    write(traj_frames_path, all_traj_frames)
                else:
                    all_traj_frames = read(traj_frames_path, ":")
                print("ntrajframes: ", len(all_traj_frames))
                
                if traj_indices_path.exists():
                    traj_indices = np.load(traj_indices_path)
                    all_traj_frames = [all_traj_frames[i] for i in traj_indices]
                    #print(traj_indices)
                print("ntrajframes: ", len(all_traj_frames), f" by {traj_period} traj_period")

                # - select
                name_path = res_dir
                sorted_path = name_path / "sorted"
                selection_params = exp_dict.get("selection", None)
                selector = create_selector(selection_params, directory=sorted_path)

                if selector:
                    # -- create dir
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
                    cur_frames = all_traj_frames
                    # TODO: add info to selected frames
                    # TODO: select based on minima (Trajectory-based Boltzmann)
                    print(f"--- Selection Method {selector.name}---")
                    #print("ncandidates: ", len(cur_frames))
                    # NOTE: there is an index map between traj_indices and selected_indices
                    cur_frames = selector.select(cur_frames)
                    #print("nselected: ", len(cur_frames))
                    #write(sorted_path/f"{selector.name}-selected-{isele}.xyz", cur_frames)

        return


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "INPUT"
    )
    parser.add_argument(
        "-nj", "--njobs", default=8, type=int
    )
    parser.add_argument(
        "--calc", action="store_true",
        help = "calculate candidates"
    )
    args = parser.parse_args()

    #input_file = "/mnt/scratch2/users/40247882/oxides/graph/NewTest/input.yaml"
    input_file = args.INPUT

    input_dict = parse_input_file(input_file)
    ae = AdsorbateEvolution(input_dict, args.njobs, args.calc)
    ae.run()
    pass