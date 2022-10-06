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

"""Online Expedition.

Add configurations and train MLIPs on-the-fly during the dynamics.

"""

class OnlineDynamicsBasedExpedition(AbstractExpedition):

    """Explore a single system with an on-the-fly trained MLIP.

    Todo:
        * Check structure RMSE for an early stopping.
        * Benchmark structures after each block.

    References:
        [Xu2021] Jiayan Xu, Xiao-Ming Cao, P.Hu J. Chem. Theory Comput. 2021, 17, 7, 4465–4476.
    
    .. _[Xu2021]:
        https://pubs.acs.org/doi/full/10.1021/acs.jctc.1c00261

    """

    name = "expedition"

    collection_params = dict(
        resdir_name = "sorted",
        init_data = None,
        traj_period = 1,
        selection_tags = ["final"]
    )

    def run(self, working_directory):
        """Run exploration."""
        nexps = len(self.explorations.keys())
        # TODO: explore many times?
        assert nexps == 1, "Online expedition only supports one at a time."

        super().run(working_directory)

        return

    def _prior_create(self, input_params: dict, *args, **kwargs) -> dict:
        """Prepare actions used through explorations."""
        actions = super()._prior_create(input_params, *args, **kwargs)

        self.pot_worker.logger = self.logger

        # TODO: check if potential is available
        driver_params = input_params["create"]["driver"]
        if self.pot_worker.potter.calc:
            driver = self.pot_worker.potter.create_driver(driver_params)
            actions["driver"] = driver
        else:
            actions["driver"] = None

        return actions
    
    def _irun(self, exp_name: str, exp_directory: Union[str,Path]) -> NoReturn:
        """Create expedition tasks and gather results."""
        # - a few info
        exp_dict = self.explorations[exp_name]
        included_systems = exp_dict.get("systems", None)
        assert len(included_systems)==1, "Online expedition only supports one system."

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
            generator, cons_text = self._read_structure(slabel)
            actions["generator"] = generator

            # - check driver
            if actions["driver"]:
                # use prepared model
                self.logger.info("Use prepared calculator+driver...")
            else:
                init_data = self.collection_params["init_data"]
                if init_data is not None:
                    self.logger.info("\n\n--- Start Initial Model ---")
                    init_frames = read(init_data, ":")
                    self.logger.info(f"Train calculator on inital dataset, {len(init_frames)}...")
                    is_trained = self._single_train(self.step_dpath, init_frames)
                    if is_trained:
                        actions["driver"] = self.pot_worker.potter.create_driver(exp_dict["create"]["driver"])
                        #print("committee: ", self.pot_worker.potter.committee)
                    else:
                        continue
                    self.logger.info("--- End Initial Model ---\n\n")
                else:
                    raise RuntimeError("Either prepared init model or init dataset should be provided.")

            # --- update cons text
            # TODO: need a unified interface here...
            actions["driver"].run_params.update(constraint=cons_text)

            # - run
            self.step_dpath = self._make_step_dir(res_dpath, "create")
            self._single_create(res_dpath, actions, ran_size=self.init_systems[slabel].get("size", 1))

        return

    def _single_create(self, res_dpath, actions, *args, **kwargs):
        """Run simulation."""
        generator = actions["generator"]
        frames = generator.run(kwargs.get("ran_size", 1))
        assert len(frames) == 1, "Online expedition only supports one structure."

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
        driver_params = driver.as_dict()
        
        # --- worker
        self.pot_worker.logger = self.logger

        self.ref_worker.directory = label_dpath
        self.ref_worker.logger = self.logger

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
            atoms = driver.run(atoms, read_exists=True, **block_run_params)
            # --- unchecked frames, and some may be labelled
            traj_fpath = collect_dpath/f"traj-b{i}.xyz"
            if not traj_fpath.exists():
                traj_frames = driver.read_trajectory()
                write(traj_fpath, traj_frames)
            else:
                traj_frames = read(traj_fpath, ":")
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
            traj_frames = self._single_collect(collect_dpath, traj_frames, actions, block_step=i)
            # - checkpoint
            # TODO: save checkpoint?
            #       current structure, current potential
            atoms.info["block"] = i
            write(check_point, atoms)
            # - train and use new driver
            if traj_frames: # selected ones
                self.logger.info(f"Found {len(traj_frames)} structures need label...")
                is_calculated = self._single_label(label_dpath, traj_frames, block_step=i)
                if is_calculated: # (label_dpath/"calculated.xyz").exists() == True
                    # - find all structures in the database
                    train_frames = []
                    for xyz_fpath in label_dpath.glob("*.xyz"):
                        xyz_frames = read(xyz_fpath, ":")
                        if xyz_frames:
                            train_frames.extend(xyz_frames)
                            self.logger.info(f"Find dataset {str(xyz_fpath)} with nframes {len(xyz_frames)}...")
                    is_trained = self._single_train(train_dpath, train_frames, block_step=i)
                    if is_trained:
                        driver = self.pot_worker.potter.create_driver(driver_params)
                    else:
                        self.logger.info("break, wait for training...")
                        break
                else:
                    self.logger.info("break, wait for labelling...")
                    break
            else:
                self.logger.info("No candidates to calculate...")

        return 

    def _single_collect(self, res_dpath, frames, actions, block_step=0,*args, **kwargs):
        """Collect results.
        
        Check whether a group of structues should be labelled
        
        """
        #for atoms in frames:
        #    print(atoms.calc.results)

        # NOTE: frames = traj_frames + labelled_frames
        selector = actions["selector"]
        selector.prefix = "traj"

        #sdirs = []
        #for p in res_dpath.iterdir():
        #    if p.is_dir():
        #        sdirs.append(p)
        #sdirs = sorted(sdirs, key=lambda x:int(x.name[1:]))
        #if sdirs:
        #    number = int(sdirs[-1].name[1:])
        #    select_dpath = res_dpath / ("s"+str(number+1))
        #else:
        #    select_dpath = res_dpath / "s0"
        select_dpath = res_dpath/("s"+str(block_step))
        select_dpath.mkdir(exist_ok=True)
        selector.directory = select_dpath

        frames_to_label = selector.select(frames)
        # TODO: check if selected frames have already been lablled
        # TODO: use database?
        write(res_dpath/f"candidates-b{block_step}.xyz", frames_to_label, append=True)

        self.logger.info(f"SELECTION: {select_dpath.name}")
        self.logger.info("confids: {}".format(" ".join([str(a.info["confid"]) for a in frames_to_label])))

        return frames_to_label
    
    def _single_label(self, res_dpath, traj_frames, block_step=0, *args, **kwargs):
        """Label candidates."""
        # - label structures
        #self.ref_worker._submit = False
        label_path = res_dpath/("l"+str(block_step))
        label_path.mkdir(exist_ok=True)
        self.ref_worker.directory = label_path

        self.ref_worker.batchsize = len(traj_frames) # TODO: custom settings?
        self.ref_worker.run(traj_frames)
        # TODO: wait here
        is_calculated = True
        calibrated_frames = self.ref_worker.retrieve()
        if calibrated_frames:
            #write(res_dpath/"calculated.xyz", calibrated_frames, append=True)
            write(res_dpath/f"calculated-b{block_step}.xyz", calibrated_frames)
        if self.ref_worker.get_number_of_running_jobs() > 0:
            is_calculated = False

        return is_calculated
    
    def _single_train(self, wdir, frames, block_step=0, *args, **kwargs):
        """Train and freeze potentials."""
        # - find train dirs
        #tdirs = []
        #for p in wdir.iterdir():
        #    if p.is_dir():
        #        tdirs.append(p)
        #tdirs = sorted(tdirs, key=lambda x:int(x.name[1:]))

        # TODO: check if finished? if not, use previous train dir
        #if tdirs:
        #    number = int(tdirs[-1].name[1:])
        #    cur_tdir = wdir / ("t"+str(number+1))
        #else:
        #    cur_tdir = wdir / "t0"
        cur_tdir = wdir / ("t"+str(block_step))

        # - train potentials
        self.logger.info(f"TRAIN at {cur_tdir.name}, {self.pot_worker.potter.train_size} models...")
        self.logger.info(f"dataset size (nframes): {len(frames)}")
        #self.pot_worker._submit = False
        self.pot_worker.directory = cur_tdir
        _ = self.pot_worker.run(frames, size=self.pot_worker.potter.train_size)
        
        # - check if training is finished
        is_trained = True
        _ = self.pot_worker.retrieve()
        if self.pot_worker.get_number_of_running_jobs() > 0:
            is_trained = False
        else:
            # - update potter's calc
            self.logger.info("UPDATE POTENTIAL...")
            self.pot_worker.potter.freeze(cur_tdir)

        return is_trained


if __name__ == "__main__":
    pass