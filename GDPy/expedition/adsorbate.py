#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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

        actions["driver"] = self.pot_worker.potter.create_driver(input_params["create"]["driver"])

        return actions

    def _single_create(self, res_dpath, actions, data, *args, **kwargs):
        """
        """
        frames = data["init_frames"]
        self.logger.info(f"number of initial structures: {len(frames)}")

        driver = actions["driver"]
        worker = self.pot_worker

        # - run
        worker.logger = self.logger
        worker.directory = self.step_dpath/"tmp_folder"
        worker.driver = driver
        worker.batchsize = len(frames)

        worker.run(frames) # BUG: re-run worker may have duplicated info in _local_slurm.json

        # - inspect
        worker.inspect()
        if worker.get_number_of_running_jobs() > 0:
            is_finished = False
        else:
            is_finished = True
        self.logger.info(f"create status: {is_finished}")

        return is_finished
    
    def _single_collect(self, res_dpath, actions, data, *args, **kwargs):
        """"""
        frames = data["init_frames"]
        self.logger.info(f"number of initial structures: {len(frames)}")

        traj_period = self.collection_params["traj_period"]

        driver = actions["driver"]
        worker = self.pot_worker

        worker.logger = self.logger
        worker.directory = res_dpath/"create"/"tmp_folder"
        worker.driver = driver
        worker.batchsize = len(frames)
        
        # - read and save trajectories
        trajectories = worker.retrieve(
            read_traj=True, traj_period=traj_period, include_first=True
        )

        #traj_frames, conv_frames = [], []
        for traj_frames in trajectories:
            # NOTE: the last frame of the trajectory, better converged...
            write(self.step_dpath/"conv_frames.xyz", traj_frames[-1], append=True)
            # NOTE: minimisation trajectories...
            write(self.step_dpath/"traj_frames.xyz", traj_frames[:-1], append=True)

        if len(worker._get_unretrieved_jobs()) > 0:
            is_collected = False
        else:
            is_collected = True
            # NOTE: sort local minima with energies
            merged_conv_frames = read(self.step_dpath/"conv_frames.xyz", ":")
            #energies = [a.get_potential_energy() for a in merged_conv_frames]
            merged_conv_frames = sorted(merged_conv_frames, key=lambda a: a.get_potential_energy())
            write(self.step_dpath/"conv_frames.xyz", merged_conv_frames)
        
        # - pass data
        if is_collected:
            merged_conv_frames = read(self.step_dpath/"conv_frames.xyz", ":")
            merged_traj_frames = read(self.step_dpath/"traj_frames.xyz", ":")

            self.logger.info(f"nconv: {len(merged_conv_frames)}")
            self.logger.info(f"ntraj: {len(merged_traj_frames)}")

            data["pot_frames_traj"] = merged_traj_frames
            data["pot_frames_conv"] = merged_conv_frames

        return is_collected


if __name__ == "__main__":
    pass