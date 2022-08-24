#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import copy

from dataclasses import dataclass
from typing import Union, Callable

import numpy as np

from ase import Atoms
from ase.io import read, write

from GDPy.expedition.abstract import AbstractExpedition

from GDPy.scheduler.factory import create_scheduler
from GDPy.selector.traj import BoltzmannMinimaSelection

from GDPy.computation.utils import read_trajectories
from GDPy.computation.worker.worker import create_worker

from GDPy.utils.command import CustomTimer


class RandomExplorer(AbstractExpedition):
    # NOTE: population-based exploration

    """
    Quasi-Random Structure Search
        ASE-GA, USPEX, AIRSS
    Algorithms
        Monte Carlo
        Basin Hopping, Simulated Annealing, Minima Hopping
        Evolutionary/Genetic Algorithm
    Steps
        create -> collect -> select -> calc -> harvest
    """

    # select params
    method_name = "GA"

    # collect parameters
    collection_params = dict(
        resdir_name = "sorted",
        traj_period = 1,
        selection_tags = ["final"],
        boltz = dict(
            fmax = 0.5, # eV
            boltzmann = 3.0, # for minima selection
            number = [16, 0.2] # number, ratio
        )
    )

    creation_params = dict(
        # - create
        #gen_fname = "gen-ads.xyz", # generated ads structures filename
        opt_fname = "opt-ads.xyz",
        opt_dname = "tmp_folder", # dir name for optimisations
        struc_prefix = "cand",
    )
    
    def _prior_create(self, input_params: dict, *args, **kwargs):
        """ some codes before creating exploratiosn of systems
            parse actions for this exploration from dict params
        """
        actions = super()._prior_create(input_params)

        create_params = input_params.get("create", None)
        task_params = copy.deepcopy(create_params["task"])
        worker_params = copy.deepcopy(create_params["worker"])
        scheduler_params = copy.deepcopy(create_params["scheduler"])

        # - task
        actions["task"] = task_params

        # NOTE: check input is valid
        # - worker
        worker = create_worker(copy.deepcopy(worker_params))
        #cur_worker_params = worker.driver.as_dict()
        #cur_worker_params.update(scheduler=params["worker"].get("scheduler", {}))
        #params["worker"].update(**cur_worker_params)
        actions["worker"] = worker

        # - scheduler
        scheduler = create_scheduler(scheduler_params)
        actions["scheduler"] = scheduler

        return actions

    def _single_create(self, res_dpath, frames, actions, *args, **kwargs):
        """ generator + driver + propagator
        """

        return

    def icreate(self, exp_name, working_directory):
        """ create explorations
        """
        exp_dict = self.explorations[exp_name]
        exp_systems = exp_dict["systems"]

        create_params = exp_dict.get("create", None)

        task_params = copy.deepcopy(create_params["task"])
        worker_params = copy.deepcopy(create_params["worker"])
        scheduler_params = copy.deepcopy(create_params["scheduler"])

        for slabel in exp_systems:
            # - prepare output directory
            res_dpath = working_directory / exp_name / slabel / "create"
            if not res_dpath.exists():
                res_dpath.mkdir(parents=True)
            
            # - prepare input
            params = copy.deepcopy(task_params)
            params["system"] = copy.deepcopy(self.init_systems[slabel].get("generator", None))
            substrate = Path(params["system"].get("substrate", None))
            if substrate:
            #    shutil.copy(substrate, res_dpath/substrate.name)
                params["system"]["substrate"] = Path(substrate).resolve()
            # TODO: check generator is valid

            # --- worker
            params["worker"] = copy.deepcopy(worker_params)
            params["worker"]["prefix"] = slabel

            constraint = self.init_systems[slabel].get("constraint", None)
            if constraint:
                params["worker"]["driver"]["run"]["constraint"] = constraint
            
            # NOTE: check input is valid
            worker = create_worker(copy.deepcopy(params["worker"]))
            cur_worker_params = worker.driver.as_dict()
            cur_worker_params.update(scheduler=params["worker"].get("scheduler", {}))
            params["worker"].update(**cur_worker_params)
            
            import yaml
            with open(res_dpath/"task.yaml", "w") as fopen:
                yaml.safe_dump(params, fopen, indent=2)
            
            # - scheduler
            #print(scheduler_params)
            scheduler = create_scheduler(scheduler_params)
            scheduler.script = res_dpath / "run.script"
            scheduler.write()
            # TODO: if submit

        return

    def icollect(self, exp_name, working_directory):
        """
        """
        exp_dict = self.explorations[exp_name]
        exp_systems = exp_dict["systems"]

        # NOTE: will the params be used many times?
        collect_params_ = copy.deepcopy(self.collection_params)
        collect_params_.update(**exp_dict.get("collect", {}))

        self.collection_params = collect_params_

        boltz_selection = BoltzmannMinimaSelection(**collect_params_["boltz"])

        actions = self._prior_create(exp_dict)

        for slabel in exp_systems:
            res_dpath = working_directory/exp_name/slabel
            create_path = res_dpath / "create"
            collect_path = self._make_step_dir(res_dpath, "collect")
            self.step_dpath = collect_path

            # TODO: add custom check whether the job is finished
            if not (create_path/"results").exists():
                print(f"{slabel} is not finished.")
                continue

            # - find results
            cand_indices = ":" # TODO: as an input?
            candidates = read(
                # TODO: a unified interface?
                create_path / "results" / "all_candidates.xyz", cand_indices
            )

            # - read trajectories
            boltz_selection.directory = self.step_dpath
            converged_indices = boltz_selection.select(candidates, ret_indices=True)
            converged_frames = [candidates[i] for i in converged_indices]
            print("nconverged: ", len(converged_frames))

            write(self.step_dpath/"converged_frames.xyz", converged_frames)

            # - final select
            self._single_collect(res_dpath, converged_frames, actions)

        return

    def _single_collect(self, res_dpath, frames, actions, *args, **kwargs):
        """"""
        traj_period = self.collection_params["traj_period"]
        print("traj_period: ", traj_period)

        # - create collect dir
        driver = actions["worker"].driver

        tmp_folder = res_dpath / "create" / self.creation_params["opt_dname"] 

        traj_dirs = []
        for i, atoms in enumerate(frames):
            confid = atoms.info["confid"]
            #print("confid: ", confid)
            traj_dir = tmp_folder / (self.creation_params["struc_prefix"]+str(confid))
            traj_dirs.append(traj_dir)
        
        # - act, retrieve trajectory frames
        merged_traj_frames = []

        traj_fpath = self.step_dpath / f"traj_frames.xyz"
        traj_ind_fpath = self.step_dpath / f"traj_indices.npy"
        with CustomTimer(name=f"collect-trajectories"):
            cur_traj_frames = read_trajectories(
                driver, traj_dirs, 
                traj_period, traj_fpath, traj_ind_fpath
            )
        merged_traj_frames.extend(cur_traj_frames)

        # - select
        selector = actions["selector"]
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