#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import logging
import copy

from pathlib import Path
from typing import NoReturn, Union

from ase.io import read, write
from ase.calculators.singlepoint import SinglePointCalculator

from GDPy import config
from GDPy.expedition.abstract import AbstractExpedition
from GDPy.computation.utils import create_single_point_calculator

""" Online Expedition
    add configurations and train MLIPs on-the-fly during
    the dynamics
"""

class OnlineDynamicsBasedExpedition(AbstractExpedition):

    """ Explore a single system with an on-the-fly trained MLIP
        Jiayan Xu, Xiao-Ming Cao, P.Hu J. Chem. Theory Comput. 2021, 17, 7, 4465–4476
    """

    name = "expedition"

    collection_params = dict(
        resdir_name = "sorted",
        traj_period = 1,
        selection_tags = ["final"],
        devi_etol = [0.02, 0.20], # eV, tolerance for atomic energy deviation
        devi_ftol = [0.05, 0.25] # eV/AA, tolerance for force deviation
    )

    def __init__(self, params: dict, potter, referee=None):
        """"""
        self._register_type_map(params) # obtain type_list or type_map

        self.explorations = params["explorations"]
        self.init_systems = params["systems"]

        self._parse_general_params(params)

        self.njobs = config.NJOBS

        # - potential and reference
        self.pot_manager = potter
        self.referee = referee # either a manager or a worker

        # - parse params
        # --- create
        # --- collect/select
        # --- label/acquire

        return
    
    def run(self, operator, working_directory):
        """"""
        nexps = len(self.explorations.keys())
        # TODO: explore many times?
        assert nexps == 1, "Online expedition only supports one at a time."

        working_directory = Path(working_directory)
        if not working_directory.exists():
            working_directory.mkdir(parents=True)
        for exp_name in self.explorations.keys():
            # note: check dir existence in sub function
            exp_directory = working_directory / exp_name
            if not exp_directory.exists():
                exp_directory.mkdir(parents=True)
            operator(exp_name, exp_directory)

        return

    def _prior_create(self, input_params: dict, *args, **kwargs) -> dict:
        """"""
        actions = super()._prior_create(input_params, *args, **kwargs)

        # TODO: check if potential is available
        driver_params = input_params["create"]["driver"]
        driver = self.pot_manager.create_driver(driver_params)
        actions["driver"] = driver

        return actions
    
    def icreate(self, exp_name: str, exp_directory: Union[str,Path]) -> NoReturn:
        """ create expedition tasks and gather results
        """
        # - a few info
        exp_dict = self.explorations[exp_name]
        included_systems = exp_dict.get("systems", None)
        assert len(included_systems)==1, "Online expedition only supports one system."

        actions = self._prior_create(exp_dict)
        print("actions: ", actions)

        # - TODO: check potential if it has uncertainty-quantification

        # - check logger
        self._init_logger(exp_directory)

        # - run over systems
        for slabel in included_systems:
            # - prepare output directory
            res_dpath = exp_directory / slabel
            if not res_dpath.exists():
                res_dpath.mkdir(parents=True)
            self.logger.info(f"===== Explore System {slabel} =====")

            # - read substrate
            self.step_dpath = self._make_step_dir(res_dpath, "init")
            frames, cons_text = self._read_structure(slabel)
            assert len(frames) == 1, "Online expedition only supports one structure."

            # - run
            self.step_dpath = self._make_step_dir(res_dpath, "create")
            self._single_create(res_dpath, frames, actions)

        return

    def _single_create(self, res_dpath, frames, actions, *args, **kwargs):
        """"""
        # - create dirs
        collect_dpath = self._make_step_dir(res_dpath, f"collect")
        self.logger.info(collect_dpath)

        label_dpath = self._make_step_dir(res_dpath, f"label")
        self.logger.info(label_dpath)

        # - check components
        # --- selector
        assert "selector" in actions.keys(), "Online needs a selector..."

        # --- driver
        driver = actions["driver"]
        #assert driver.name == "ase", "Online expedition only supports ase dynamics."
        driver.directory = self.step_dpath / "tmp_folder"
        # driver.calc.directory = str(self.step_dpath/"driver")
        
        # --- worker
        self.referee.directory = label_dpath

        # --- steps and intervals
        traj_period = self.collection_params["traj_period"]
        dump_period = driver.get("dump_period")
        block_run_params = copy.deepcopy(driver.run_params)
        block_steps = traj_period*dump_period
        block_run_params["steps"] = block_steps
        tot_steps = driver.get("steps")
        nblocks = int(tot_steps/block_steps) + 1
        self.logger.info(f"dump_period: {dump_period} traj_period: {traj_period}")
        self.logger.info(f"tot: {tot_steps} block: {block_steps} nblocks: {nblocks}")

        # - from the scratch
        atoms = copy.deepcopy(frames[0])

        # TODO: read check_point
        check_point = self.step_dpath / "check_point.xyz"

        for i in range(nblocks):
            self.logger.info(f"\n\n----- step {i:8d} -----")
            # - NOTE: run a block, dyn, without bias (plumed)
            #dynamics.step()
            atoms = driver.run(atoms, **block_run_params)
            # - create a copy of atoms
            #spc_atoms = atoms.copy()
            #calc = SinglePointCalculator(spc_atoms)
            #calc.results = copy.deepcopy(atoms.calc.results)
            #calc.name = self.pot_manager.name
            #spc_atoms.calc = calc
            #spc_atoms.info["step"] = i
            #for k, v in spc_atoms.calc.results.items(): # TODO: a unified interface?
            #    if "devi" in k:
            #        spc_atoms.info[k] = v
            #traj_frames.append(spc_atoms)
            traj_frames = driver.read_trajectory() # unchecked frames, and some may be labelled
            confids = range(i*block_steps,(i+1)*block_steps,dump_period)
            for a, confid in zip(traj_frames,confids):
                a.info["confid"] = confid
            # - collect, select, label
            self.logger.info(f"num_traj_frames: {len(traj_frames)}")
            traj_frames = self._single_collect(collect_dpath, traj_frames, actions)
            # - checkpoint
            # TODO: save checkpoint?
            #       current structure, current potential
            write(check_point, atoms)
            # save data
            # - train and use new driver
            if traj_frames:
                self.logger.info(f"Found {len(traj_frames)} structures need label...")
                self._single_label(label_dpath, traj_frames)
            else:
                self.logger.info("No candidates to calculate...")
            # - empty structure
            traj_frames = [] # remove structures being calculated
        
        # - restart from a run

        return 

    def _single_collect(self, res_dpath, frames, actions, *args, **kwargs):
        """ check whether a group of structues should be labelled
        """
        #for atoms in frames:
        #    print(atoms.calc.results)

        # NOTE: frames = traj_frames + labelled_frames
        selector = actions["selector"]
        selector.prefix = "traj"

        sdirs = []
        for p in res_dpath.iterdir():
            if p.is_dir():
                sdirs.append(p)
        sdirs = sorted(sdirs, key=lambda x:int(x.name[1:]))

        if sdirs:
            number = int(sdirs[-1].name[1:])
            select_dpath = res_dpath / ("s"+str(number+1))
        else:
            select_dpath = res_dpath / "s0"
        self.logger.info(f"SELECTION: {select_dpath.name}")
        select_dpath.mkdir()
        selector.directory = select_dpath

        frames_to_label = selector.select(frames)
        # TODO: check if selected frames have already been lablled
        # TODO: use database?
        write(res_dpath/"candidates.xyz", frames_to_label, append=True)

        return frames_to_label
    
    def _single_label(self, res_dpath, traj_frames, *args, **kwargs):
        """"""
        # - label structures
        self.referee.run(traj_frames)
        # TODO: wait here
        is_calculated = False
        while self.referee.get_number_of_running_jobs() > 0:
            calibrated_frames = self.referee.retrieve()
            if calibrated_frames:
                write(res_dpath/"calculated.xyz", calibrated_frames, append=True)
        else:
            is_calculated = True
        # - train potentials
        is_trained = self.pot_manager.train(res_dpath/"calculataed.xyz")

        return traj_frames


if __name__ == "__main__":
    pass