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
        frames = data["pot_frames"]
        self.logger.info(f"number of initial structures: {len(frames)}")

        driver = actions["driver"]
        worker = self.pot_worker

        # - run
        worker.logger = self.logger
        worker.directory = self.step_dpath/"tmp_folder"
        worker.driver = driver
        worker.batchsize = len(frames)

        worker.run(frames)

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
        frames = data["pot_frames"]
        self.logger.info(f"number of initial structures: {len(frames)}")

        traj_period = self.collection_params["traj_period"]

        driver = actions["driver"]
        worker = self.pot_worker

        worker.logger = self.logger
        worker.directory = res_dpath/"create"/"tmp_folder"
        worker.driver = driver
        worker.batchsize = len(frames)
        
        # TODO: save converged frames (last frame of each trajectory?)
        traj_frames = worker.retrieve(
            read_traj=True, traj_period=traj_period, include_first=True
        )
        write(self.step_dpath/"traj_frames.xyz", traj_frames, append=True)

        if len(worker._get_unretrieved_jobs()) > 0:
            is_collected = False
        else:
            is_collected = True
        
        # - pass data
        if is_collected:
            merged_traj_frames = read(self.step_dpath/"traj_frames.xyz", ":")
            data["pot_frames"] = merged_traj_frames

        return is_collected


if __name__ == "__main__":
    pass