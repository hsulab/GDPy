#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
from collections import Counter
from typing import List

import numpy as np

from ase import Atoms
from ase.io import read, write

from GDPy.expedition.abstract import AbstractExpedition
from GDPy.computation.utils import read_trajectories

from GDPy.utils.command import CustomTimer


class AdsorbateEvolution(AbstractExpedition):

    name = "ads"

    creation_params = dict(
        # - create
        #gen_fname = "gen-ads.xyz", # generated ads structures filename
        opt_fname = "opt-ads.xyz",
        opt_dname = "tmp_folder", # dir name for optimisations
        struc_prefix = "cand",
    )

    collection_params = dict(
        traj_period = 1,
        selection_tags = ["converged", "traj"]
    )

    parameters = dict()
    
    def _prior_create(self, input_params):
        """ some codes before creating exploratiosn of systems
            parse actions for this exploration from dict params
        """
        actions = super()._prior_create(input_params)

        actions["driver"] = self.pot_manager.create_driver(input_params["create"]["driver"])

        return actions

    def _single_create(self, res_dpath, frames, actions):
        """
        """
        driver = actions["driver"]

        # - run exploration
        cur_frames = frames
        act_outpath = self.step_dpath / self.creation_params["opt_fname"]
        tmp_folder = self.step_dpath / self.creation_params["opt_dname"]
        if not tmp_folder.exists():
            tmp_folder.mkdir()
        new_frames = []
        for i, atoms in enumerate(cur_frames): 
            confid = atoms.info["confid"] # for candid
            driver.directory = tmp_folder/(self.creation_params["struc_prefix"]+str(confid))
            # TODO: check existed results before running, lammps works
            new_atoms = driver.run(atoms, read_exists=True, extra_info=dict(confid=confid))
            #print(new_atoms.info["confid"])
            new_frames.append(new_atoms)
        cur_frames = new_frames
        write(act_outpath, cur_frames) # NOTE: update output frames

        return cur_frames
    
    def _single_collect(self, res_dpath, frames, actions, *args, **kwargs):
        """"""
        traj_period = self.collection_params["traj_period"]

        # - create collect dir
        driver = actions["driver"]

        tmp_folder = res_dpath / "create" / self.creation_params["opt_dname"] 

        traj_dirs = []
        for i, atoms in enumerate(frames):
            confid = atoms.info["confid"]
            traj_dir = tmp_folder / (self.creation_params["struc_prefix"]+str(confid))
            traj_dirs.append(traj_dir)
        
        # - act, retrieve trajectory frames
        merged_traj_frames = []

        traj_fpath = self.step_dpath / f"traj_frames.xyz"
        traj_ind_fpath = self.step_dpath / f"traj_indices.npy"
        with CustomTimer(name=f"collect-trajectories"):
            cur_traj_frames = read_trajectories(
                driver, traj_dirs, traj_period, 
                traj_fpath, traj_ind_fpath
            )
        merged_traj_frames.extend(cur_traj_frames)

        # - select
        selector = actions.get("selector", None)
        if selector:
            # -- create dir
            select_dpath = self._make_step_dir(res_dpath, "select")
            # -- perform selections
            cur_frames = merged_traj_frames
            # TODO: add info to selected frames
            # TODO: select based on minima (Trajectory-based Boltzmann)
            print(f"--- Selection Method {selector.name}---")
            selector.prefix = "traj"
            selector.directory = select_dpath
            #print("ncandidates: ", len(cur_frames))
            # NOTE: there is an index map between traj_indices and selected_indices
            cur_frames = selector.select(cur_frames)
            #print("nselected: ", len(cur_frames))
            #write(sorted_path/f"{selector.name}-selected-{isele}.xyz", cur_frames)
        else:
            print("No selector available...")
        
        return


if __name__ == "__main__":
    pass