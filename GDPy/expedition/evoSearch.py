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

from GDPy.scheduler import create_scheduler
from GDPy.selector.traj import BoltzmannMinimaSelection

from GDPy.computation.utils import read_trajectories
from GDPy.computation.worker.command import CommandWorker
from GDPy.potential.register import create_potter

from GDPy.utils.command import CustomTimer


class EvolutionaryExpedition(AbstractExpedition):

    """Quasi-Random Structure Search.

    Current method is the genetic algorithm implemented in ASE. The general workflow
    is (create -> collect -> select -> calc -> harvest). Other methods may utilise
    this workflow as well, nanmely, Monte Carlo, Basin Hopping, Simulated Annealing, 
    and Minima Hopping.

    NOTE: 
        This is only for the population-based exploration now.

    TODO:
        * support USPEX and AIRSS

    """

    # select params
    name = "evo"

    # collect parameters
    collection_params = dict(
        traj_period = 1,
        selection_tags = ["converged", "traj"],
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
        scheduler_params = copy.deepcopy(create_params["scheduler"])

        # - task
        actions["task"] = task_params

        # NOTE: check input is valid
        # - worker
        actions["worker"] = self.pot_worker

        # TODO: check constraint
        actions["driver"] = self.pot_worker.driver

        # - scheduler
        scheduler = create_scheduler(scheduler_params)
        actions["scheduler"] = scheduler

        actions["command_worker"] = CommandWorker(scheduler)

        # - collect
        boltz_selection = BoltzmannMinimaSelection(**self.collection_params["boltz"])
        actions["boltz"] = boltz_selection

        return actions

    def _single_create(self, res_dpath, actions, data, *args, **kwargs):
        """ generator + driver + propagator
        """
        # - prepare input
        params = copy.deepcopy(actions["task"])

        # TODO: check generator should not be a direct one
        from GDPy.builder.direct import DirectGenerator
        assert not isinstance(actions["generator"], DirectGenerator), "Generator for EvoSearch should not be direct."
        params["system"] = actions["generator"].as_dict()
        
        import yaml
        with open(self.step_dpath/"task.yaml", "w") as fopen:
            yaml.safe_dump(params, fopen, indent=2)

        # - worker
        worker = actions["worker"]
        worker_params = {}
        worker_params["potential"] = worker.potter.as_dict()
        worker_params["driver"] = worker.driver.as_dict()

        with open(self.step_dpath/"worker.yaml", "w") as fopen:
            yaml.safe_dump(worker_params, fopen, indent=2)
        
        command_worker = actions["command_worker"]
        command_worker.logger = self.logger
        command_worker.directory = self.step_dpath
        command_worker.run()

        command_worker.inspect() # TODO: add convergence check?
        if command_worker.get_number_of_running_jobs() > 0:
            is_finished = False
        else:
            is_finished = True
        
        return is_finished

    def _single_collect(self, res_dpath, actions, data, *args, **kwargs):
        """"""
        traj_period = self.collection_params["traj_period"]
        self.logger.info(f"traj_period: {traj_period}")

        # TODO: add custom check whether the job is finished
        create_path = res_dpath/"create"
        #if not (create_path/"results").exists():
        #    print(f"{slabel} is not finished.")
        #    continue

        # - find results
        cand_indices = ":" # TODO: as an input?
        candidates = read(
            # TODO: a unified interface?
            create_path / "results" / "all_candidates.xyz", cand_indices
        )

        # - read trajectories
        boltz_selection = actions["boltz"]
        boltz_selection.directory = self.step_dpath
        converged_indices = boltz_selection.select(candidates, ret_indices=True)
        converged_frames = [candidates[i] for i in converged_indices]
        self.logger.info(f"nconverged: {len(converged_frames)}")

        write(self.step_dpath/"converged_frames.xyz", converged_frames)

        # - create collect dir
        tmp_folder = create_path / self.creation_params["opt_dname"] 

        traj_dirs = []
        for i, atoms in enumerate(converged_frames):
            confid = atoms.info["confid"]
            #print("confid: ", confid)
            traj_dir = tmp_folder / (self.creation_params["struc_prefix"]+str(confid))
            traj_dirs.append(traj_dir)
        
        # - act, retrieve trajectory frames
        worker = actions["worker"]
        worker.logger = self.logger

        traj_fpath = self.step_dpath / "traj_frames.xyz"

        if not traj_fpath.exists():
            self.logger.info("Used worker to read trajs...")
            assert worker.batchsize == 1, "Worker must have a batchsize of one."
            merged_traj_frames = worker._read_results(
                traj_dirs, read_traj=True, 
                traj_period=traj_period, include_first=False, include_last=True
            )
            write(traj_fpath, merged_traj_frames)
        else:
            self.logger.info("Used cached traj_frames...")
            merged_traj_frames = read(traj_fpath, ":")

        # - pass data
        data["pot_frames"] = merged_traj_frames

        is_collected = True
        
        return is_collected
    

if __name__ == "__main__":
    pass