#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import logging
import copy

from pathlib import Path
from typing import NoReturn, Union

from ase.io import read, write
from ase.constraints import FixAtoms
from ase.calculators.singlepoint import SinglePointCalculator

from GDPy import config
from GDPy.expedition.abstract import AbstractExpedition
from GDPy.computation.utils import create_single_point_calculator
from GDPy.builder.constraints import parse_constraint_info

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
        init_data = None,
        traj_period = 1,
        selection_tags = ["final"]
    )

    def __init__(self, params: dict, potter, referee=None):
        """"""
        self._register_type_map(params) # obtain type_list or type_map

        self.explorations = params["explorations"]
        self.init_systems = params["systems"]

        self._parse_general_params(params)

        self.njobs = config.NJOBS

        # - potential and reference
        self.potter = potter
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
        if self.potter.potter.calc:
            driver = self.potter.potter.create_driver(driver_params)
            actions["driver"] = driver
        else:
            actions["driver"] = None

        return actions
    
    def icreate(self, exp_name: str, exp_directory: Union[str,Path]) -> NoReturn:
        """ create expedition tasks and gather results
        """
        # - a few info
        exp_dict = self.explorations[exp_name]
        included_systems = exp_dict.get("systems", None)
        assert len(included_systems)==1, "Online expedition only supports one system."

        # - check logger
        self._init_logger(exp_directory)

        # TODO: check if potential is available
        actions = self._prior_create(exp_dict)

        # --- selector
        # - TODO: check potential if it has uncertainty-quantification
        assert "selector" in actions.keys(), "Online needs a selector..."

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

            # - check driver
            if actions["driver"]:
                # use prepared model
                self.logger.info("Use prepared calculator+driver...")
            else:
                init_data = self.collection_params["init_data"]
                if init_data is not None:
                    init_frames = read(init_data, ":")
                    self.logger.info(f"Train calculator on inital dataset, {len(init_frames)}...")
                    self._single_train(self.step_dpath, init_frames)
                    actions["driver"] = self.potter.potter.create_driver(exp_dict["create"]["driver"])
                else:
                    raise RuntimeError("Either prepared init model or init dataset should be provided.")

            # - run
            self.step_dpath = self._make_step_dir(res_dpath, "create")
            self._single_create(res_dpath, frames, actions)

        return

    def _single_create(self, res_dpath, frames, actions, *args, **kwargs):
        """"""
        # - create dirs
        collect_dpath = self._make_step_dir(res_dpath, f"collect")
        label_dpath = self._make_step_dir(res_dpath, f"label")
        train_dpath = self._make_step_dir(res_dpath, f"train")

        # --- dynamics trajectory path
        tmp_folder = self.step_dpath / "tmp_folder"
        if not tmp_folder.exists():
            tmp_folder.mkdir()

        # - check components
        # --- driver
        driver = actions["driver"]
        driver_params = driver.as_dict()["driver"]
        
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

        # - read check_point
        check_point = self.step_dpath / "check_point.xyz"
        if check_point.exists():
            # - from the scratch
            atoms = read(check_point)
            start_block = atoms.info["block"]
        else:
            atoms = copy.deepcopy(frames[0])
            start_block = 0
        self.logger.info(f"start_block: {start_block}")

        # --- add init data to total dataset
        if start_block == 0:
            init_data = self.collection_params["init_data"]
            if init_data is not None:
                init_frames = read(init_data, ":")
                write(label_dpath/"calculated.xyz", init_frames)
                self.logger.info(f"ADD init dataset, {len(init_frames)}...")

        # - run dynamics
        for i in range(start_block,nblocks):
            self.logger.info(f"\n\n----- block {i:8d} -----")
            # - NOTE: run a block, dyn, without bias (plumed)
            # - NOTE: overwrite existed data
            driver.directory = tmp_folder / ("b"+str(i))
            # TODO !!! if restart, how about this part? !!!
            #          it's ok for deterministic selection but ...
            atoms = driver.run(atoms, read_exists=True, **block_run_params)
            # --- unchecked frames, and some may be labelled
            traj_frames = driver.read_trajectory()
            confids = range(i*block_steps,(i+1)*block_steps+1,dump_period) # NOTE: include last
            for a, confid in zip(traj_frames,confids):
                a.info["confid"] = confid
                # add constraint TODO: move this part to driver?
                constraint = block_run_params.get("constraint", None)
                mobile_indices, frozen_indices = parse_constraint_info(atoms, constraint, ret_text=False)
                if frozen_indices:
                    a.set_constraint(FixAtoms(indices=frozen_indices))
                # NOTE: ase does not remove com 3 dof while lammps does
                self.logger.info(
                    f"step: {confid:>8d} temp: {a.get_temperature():>12.4f} " + 
                    f"ke: {a.get_kinetic_energy():>12.4f} pe: {a.get_potential_energy():>12.4f}"
                )
            # - collect, select, label
            self.logger.info(f"num_traj_frames: {len(traj_frames)}")
            traj_frames = self._single_collect(collect_dpath, traj_frames, actions)
            # - checkpoint
            # TODO: save checkpoint?
            #       current structure, current potential
            atoms.info["block"] = i
            write(check_point, atoms)
            # - train and use new driver
            if traj_frames: # selected ones
                self.logger.info(f"Found {len(traj_frames)} structures need label...")
                is_calculated = self._single_label(label_dpath, traj_frames)
                if is_calculated: # (label_dpath/"calculated.xyz").exists() == True
                    train_frames = read(label_dpath/"calculated.xyz", ":")
                    is_trained = self._single_train(train_dpath, train_frames)
                    if is_trained:
                        driver = self.potter.potter.create_driver(driver_params)
                    else:
                        self.logger.info("break, wait for training...")
                        break
                else:
                    self.logger.info("break, wait for labelling...")
                    break
            else:
                self.logger.info("No candidates to calculate...")

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
        select_dpath.mkdir()
        selector.directory = select_dpath

        frames_to_label = selector.select(frames)
        # TODO: check if selected frames have already been lablled
        # TODO: use database?
        write(res_dpath/"candidates.xyz", frames_to_label, append=True)

        self.logger.info(f"SELECTION: {select_dpath.name}")
        self.logger.info("confids: {}".format(" ".join([str(a.info["confid"]) for a in frames_to_label])))

        return frames_to_label
    
    def _single_label(self, res_dpath, traj_frames, *args, **kwargs):
        """"""
        # - label structures
        #self.referee._submit = False
        self.referee.run(traj_frames)
        # TODO: wait here
        is_calculated = True
        calibrated_frames = self.referee.retrieve()
        if calibrated_frames:
            write(res_dpath/"calculated.xyz", calibrated_frames, append=True)
        if self.referee.get_number_of_running_jobs() > 0:
            is_calculated = False

        return is_calculated
    
    def _single_train(self, wdir, frames, *args, **kwargs):
        """"""
        tdirs = []
        for p in wdir.iterdir():
            if p.is_dir():
                tdirs.append(p)
        sdirs = sorted(tdirs, key=lambda x:int(x.name[1:]))

        if tdirs:
            number = int(sdirs[-1].name[1:])
            cur_tdir = wdir / ("t"+str(number+1))
        else:
            cur_tdir = wdir / "t0"
        # - train potentials
        self.logger.info(f"TRAIN at {cur_tdir.name}, {self.potter.potter.train_size} models...")
        self.logger.info(f"dataset size (nframes): {len(frames)}")
        #self.potter._submit = False
        self.potter.directory = cur_tdir
        _ = self.potter.run(frames, size=self.potter.potter.train_size)
        
        # - check if training is finished
        is_trained = True
        _ = self.potter.retrieve()
        if self.potter.get_number_of_running_jobs() > 0:
            is_trained = False
        else:
            # - update potter's calc
            self.logger.info("UPDATE POTENTIAL...")
            self.potter.potter.freeze(cur_tdir)

        return is_trained


if __name__ == "__main__":
    pass