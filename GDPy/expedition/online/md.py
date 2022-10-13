#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import logging
import copy

from pathlib import Path
from typing import NoReturn, Union

import numpy as np

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
            cons_text = self._read_structure(slabel, actions)

            init_frame_path = self.step_dpath / "init.xyz" 
            generator = actions["generator"]
            init_frames = generator.run(
                ran_size=self.init_systems[slabel].get("size", 1)
            )
            if not init_frame_path.exists():
                write(
                    init_frame_path, init_frames, columns=["symbols", "positions", "move_mask"]
                )
            else:
                # TODO: assert current init_frames is the same as the cached one
                pass

            data = dict(
                init_frames = init_frames, # initial structure
                pot_frames = [], # potential-calculated traj frames
                selected_frames = [], # selected traj frames
                calibrated_frames = [] # reference calculated frames
            )

            # - check driver
            if actions["driver"]:
                # use prepared model
                self.logger.info("Use prepared calculator+driver...")
            else:
                init_data = self.collection_params["init_data"]
                if init_data is not None:
                    self.logger.info("\n\n--- Start Initial Model ---")
                    init_train_frames = read(init_data, ":")
                    self.logger.info(f"Train calculator on inital dataset, {len(init_train_frames)}...")
                    is_trained = self._single_train(self.step_dpath, init_train_frames)
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
            self._single_create(res_dpath, actions, data)

        # - report
        self.logger.info("FINISHED...")

        return

    def _single_create(self, res_dpath, actions, data, *args, **kwargs):
        """Run simulation."""
        frames = data["init_frames"]
        assert len(frames) == 1, "Online expedition only supports one structure."

        # - create dirs
        collect_dpath = self._make_step_dir(res_dpath, f"collect")
        select_dpath = self._make_step_dir(res_dpath, f"select")
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

        # --- parse constraint
        constraint = block_run_params.get("constraint", None)
        mobile_indices, frozen_indices = parse_constraint_info(atoms, constraint, ret_text=False)

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
            # - TODO: Use worker here...
            driver.directory = tmp_folder / ("b"+str(i))
            atoms = driver.run(atoms, read_exists=True, **block_run_params)

            # --- checkpoint
            # TODO: save checkpoint?
            #       current structure, current potential
            atoms.info["block"] = i
            write(check_point, atoms, columns=["symbols", "positions", "move_mask"])

            # - collect
            # --- unchecked frames, and some may be labelled
            self._single_collect(res_dpath, actions, data, block_step=i)
            
            # --- output MD information
            traj_frames = data["pot_frames"]
            confids = range(i*block_steps,(i+1)*block_steps+1,dump_period) # NOTE: include last
            for a, confid in zip(traj_frames,confids):
                a.info["confid"] = confid
                # add constraint TODO: move this part to driver?
                if frozen_indices:
                    a.set_constraint(FixAtoms(indices=frozen_indices))
                # NOTE: ase does not remove com 3 dof while lammps does
                self.logger.info(
                    f"step: {confid:>8d} temp: {a.get_temperature():>12.4f} " + 
                    f"ke: {a.get_kinetic_energy():>12.4f} pe: {a.get_potential_energy():>12.4f}"
                )
            
            # --- calculate RMSD for an early stopping or restart from the initial structure.
            rms_dis = np.sqrt(np.sum(np.square(traj_frames[0].positions-traj_frames[-1].positions)))
            self.logger.info(f"Root Mean Square of Displacement (RMSE) is {rms_dis:>12.4f}.")

            # - select
            self._single_select(res_dpath, actions, data, block_step=i)

            # - update potential
            is_calculated = self._single_label(res_dpath, actions, data, block_step=i)
            if is_calculated: # (label_dpath/"calculated.xyz").exists() == True
                # --- benchmark before learning
                self._single_benchmark(res_dpath, actions, data, block_step=i)
                # --- train
                is_trained = self._single_train(res_dpath, actions, data, block_step=i)
                if is_trained:
                    driver = self.pot_worker.potter.create_driver(driver_params)
                else:
                    self.logger.info("break, wait for training...")
                    break
                # --- benchmark after learning
            else:
                self.logger.info("break, wait for labelling...")
                break

        return 

    def _single_collect(self, res_dpath, actions, data, block_step=0,*args, **kwargs) -> None:
        """Collect results.
        
        Check whether a group of structues should be labelled
        
        """
        collect_dpath = res_dpath / "collect"
        driver = actions["driver"]

        # --- unchecked frames, and some may be labelled
        traj_fpath = collect_dpath/f"traj-b{block_step}.xyz"
        if not traj_fpath.exists():
            traj_frames = driver.read_trajectory()
            write(traj_fpath, traj_frames)
        # NOTE: To make atoms.info has a same order, we always read the outpufile,
        #       which is important for md5 check in the worker.
        traj_frames = read(traj_fpath, ":")

        data["pot_frames"] = traj_frames # potential-calculated frames

        return 
    
    def _single_select(self, res_dpath, actions, data, block_step=0, *args, **kwargs):
        """Select frames."""
        select_dpath = res_dpath/"select"
        frames = data["pot_frames"]

        # NOTE: frames = traj_frames + labelled_frames
        selector = actions["selector"]
        selector.prefix = "traj"

        cur_select_dpath = select_dpath/("s"+str(block_step))
        cur_select_dpath.mkdir(exist_ok=True)
        selector.directory = cur_select_dpath

        frames_to_label = selector.select(frames)

        self.logger.info(f"SELECTION: {select_dpath.name}")
        self.logger.info("confids: {}".format(" ".join([str(a.info["confid"]) for a in frames_to_label])))

        data["selected_frames"] = frames_to_label
        #write(res_dpath/f"candidates-b{block_step}.xyz", frames_to_label, append=True)

        return
    
    def _single_label(self, res_dpath, actions, data, block_step=0, *args, **kwargs) -> bool:
        """Label candidates.

        A file with calculated structures would be written to label path.
        
        """
        selected_frames = data["selected_frames"]
        nselected = len(selected_frames)
        self.logger.info(f"Found {nselected} structures need label...")
        if nselected == 0:
            return True

        # - label structures
        #self.ref_worker._submit = False
        label_path = res_dpath/"label"/("l"+str(block_step))
        label_path.mkdir(exist_ok=True)
        self.ref_worker.directory = label_path

        self.ref_worker.batchsize = len(selected_frames) # TODO: custom settings?
        self.ref_worker.run(selected_frames)

        # NOTE: wait here
        is_calculated = True
        calibrated_frames = self.ref_worker.retrieve()
        if calibrated_frames:
            #write(res_dpath/"calculated.xyz", calibrated_frames, append=True)
            write(res_dpath/"label"/f"calculated-b{block_step}.xyz", calibrated_frames)
            data["calibrated_frames"] = calibrated_frames
        if self.ref_worker.get_number_of_running_jobs() > 0:
            is_calculated = False

        return is_calculated
    
    def _single_train(self, res_dpath, actions, data, block_step=0, *args, **kwargs):
        """Train and freeze potentials."""
        # - find all structures in the database
        label_dpath = res_dpath/"label"
        train_frames = []
        for xyz_fpath in label_dpath.glob("*.xyz"):
            xyz_frames = read(xyz_fpath, ":")
            if xyz_frames:
                train_frames.extend(xyz_frames)
                self.logger.info(f"Find dataset {str(xyz_fpath)} with nframes {len(xyz_frames)}...")
        # TODO: benchmark train frames

        # - train
        cur_tdir = res_dpath/"train"/("t"+str(block_step))

        # - train potentials
        self.logger.info(f"TRAIN at {cur_tdir.name}, {self.pot_worker.potter.train_size} models...")
        self.logger.info(f"dataset size (nframes): {len(train_frames)}")
        #self.pot_worker._submit = False
        self.pot_worker.directory = cur_tdir
        _ = self.pot_worker.run(train_frames, size=self.pot_worker.potter.train_size)
        
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

    def _single_benchmark(self, res_dpath, actions, data, block_step=0, *args, **kwargs):
        """Benchmark potential at i-block with reference structures at i-block.
        """
        selected_frames = data["selected_frames"]
        calibrated_frames = data["calibrated_frames"]
        if not calibrated_frames: # if empty
            calibrated_frames = read(res_dpath/"label"/f"calculated-b{block_step}.xyz", ":")
            data["calibrated_frames"] = calibrated_frames
        nframes = len(selected_frames)
        assert nframes == len(calibrated_frames), "Number of frames must be equal."

        # - stat
        from sklearn.metrics import mean_absolute_error, mean_squared_error

        # --- energy
        pot_energies = [a.get_potential_energy() for a in selected_frames]
        ref_energies = [a.get_potential_energy() for a in calibrated_frames]

        mae = mean_absolute_error(ref_energies, pot_energies)
        rmse = mean_squared_error(ref_energies, pot_energies, squared=False)
        self.logger.info(f"Total Energy MAE: {mae:>12.4f} RMSE: {rmse:>12.4f}")

        # --- forces
        chemical_symbols = selected_frames[0].get_chemical_symbols()
        type_list = list(set(chemical_symbols))
        pot_force_map = {k: [] for k in type_list}
        ref_force_map = {k: [] for k in type_list}
        for i in range(nframes):
            pot_forces = selected_frames[i].get_forces().copy().tolist()
            ref_forces = calibrated_frames[i].get_forces().copy().tolist()
            for sym, pot_fxyz, ref_fxyz in zip(chemical_symbols, pot_forces, ref_forces):
                pot_force_map[sym].extend(pot_fxyz)
                ref_force_map[sym].extend(ref_fxyz)

        for sym in type_list:
            mae = mean_absolute_error(ref_force_map[sym], pot_force_map[sym])
            rmse = mean_squared_error(ref_force_map[sym], pot_force_map[sym], squared=False)
            self.logger.info(f"{sym:>4s} Force MAE: {mae:>12.4f} RMSE: {rmse:>12.4f}")

        return


if __name__ == "__main__":
    pass