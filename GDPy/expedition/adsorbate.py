#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
from pathlib import Path
from collections import Counter
from typing import List

import shutil
import warnings

import numpy as np

from ase import Atoms
from ase.io import read, write

from GDPy.utils.command import parse_input_file

from .abstract import AbstractExplorer
from GDPy.computation.driver import AbstractDriver

from GDPy.builder.adsorbate import StructureGenerator, AdsorbateGraphGenerator

from GDPy.utils.command import CustomTimer


class AdsorbateEvolution(AbstractExplorer):

    name = "ads"

    creation_params = dict(
        # - create
        action_order = ["adsorption", "dynamics"],
        gen_fname = "gen-ads.xyz", # generated ads structures filename
        opt_fname = "opt-ads.xyz",
        opt_dname = "tmp_folder", # dir name for optimisations
        struc_prefix = "cand",
        # - collect
        traj_period = 1
    )

    collection_params = dict(
        resdir_name = "sorted",
        selection_tags = ["converged", "traj"]
    )

    parameters = dict()

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
    
    def _prior_create(self, input_params):
        """ some codes before creating exploratiosn of systems
            parse actions for this exploration from dict params
        """
        selector = super()._prior_create(input_params)

        action_dict = {}
        for act_name, act_params in input_params["create"].items():
            #print(act_Iname, act_params)
            if act_name == "adsorption":
                action = AdsorbateGraphGenerator(act_params["composition"], act_params["graph"], Path.cwd())
            elif act_name == "dynamics":
                action = self.pot_manager.create_worker(act_params)
            else:
                pass
            action_dict[act_name] = action

        return action_dict, selector

    def _single_create(self, res_dpath, frames, cons_text, actions):
        """
        """
        # - create collect dir
        create_dpath = res_dpath / "create"
        if create_dpath.exists():
            if self.ignore_exists:
                #warnings.warn("sorted_path removed in %s" %res_dpath, UserWarning)
                shutil.rmtree(create_dpath)
                create_dpath.mkdir()
            else:
                return
        else:
            create_dpath.mkdir()
        
        # - run exploration
        cur_frames = frames
        for act_name in self.creation_params["action_order"]:
            action = actions.get(act_name, None)
            if action is None:
                continue
            if isinstance(action, StructureGenerator):
                action.directory = create_dpath # NOTE: set new path
                act_outpath = create_dpath / self.creation_params["gen_fname"]
                if not act_outpath.exists():
                    cur_frames = action.run(cur_frames)
                else:
                    cur_frames = read(act_outpath, ":")
            elif isinstance(action, AbstractDriver):
                act_outpath = create_dpath / self.creation_params["opt_fname"]
                tmp_folder = create_dpath / self.creation_params["opt_dname"]
                if not tmp_folder.exists():
                    tmp_folder.mkdir()
                new_frames = []
                for i, atoms in enumerate(cur_frames): 
                    confid = atoms.info["confid"]
                    #print(confid)
                    action.set_output_path(tmp_folder/(self.creation_params["struc_prefix"]+str(confid)))
                    # TODO: check existed results before running, lammps works
                    new_atoms = action.run(atoms, extra_info=dict(confid=confid), constraint=cons_text)
                    #print(new_atoms.info["confid"])
                    new_frames.append(new_atoms)
                cur_frames = new_frames
            else:
                pass
            write(act_outpath, cur_frames) # NOTE: update output frames

        return cur_frames
    
    def _single_collect(self, res_dpath, frames, cons_text, actions, selector, *args, **kwargs):
        """"""
        traj_period = self.creation_params["traj_period"]

        # - create collect dir
        collect_path = res_dpath / "collect"
        if collect_path.exists():
            if self.ignore_exists:
                #warnings.warn("sorted_path removed in %s" %res_dpath, UserWarning)
                shutil.rmtree(collect_path)
                collect_path.mkdir()
            else:
                pass
        else:
            collect_path.mkdir()
        
        create_dpath = res_dpath / "create"

        # - act, retrieve trajectory frames
        with CustomTimer("collect"):
            from GDPy.computation.dynamics import read_trajectories
            action = actions["dynamics"]
            all_traj_frames = read_trajectories(
                action, create_dpath / self.creation_params["opt_dname"], traj_period, 
                collect_path/"traj_frames.xyz", collect_path/"traj_indices.npy",
                create_dpath/self.creation_params["opt_fname"]
            )

        # - select
        sorted_path = res_dpath / "sorted"

        if selector:
            # -- create dir
            if sorted_path.exists():
                if self.ignore_exists:
                    warnings.warn("sorted_path removed in %s" %res_dpath, UserWarning)
                    shutil.rmtree(sorted_path)
                    sorted_path.mkdir()
                else:
                    warnings.warn("sorted_path exists in %s" %res_dpath, UserWarning)
                    return
            else:
                sorted_path.mkdir()
            # -- perform selections
            cur_frames = all_traj_frames
            # TODO: add info to selected frames
            # TODO: select based on minima (Trajectory-based Boltzmann)
            print(f"--- Selection Method {selector.name}---")
            selector.directory = sorted_path
            #print("ncandidates: ", len(cur_frames))
            # NOTE: there is an index map between traj_indices and selected_indices
            cur_frames = selector.select(cur_frames)
            #print("nselected: ", len(cur_frames))
            #write(sorted_path/f"{selector.name}-selected-{isele}.xyz", cur_frames)
        else:
            print("No selector available...")
        
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