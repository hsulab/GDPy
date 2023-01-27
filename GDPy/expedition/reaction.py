#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

from typing import NoReturn

from ase.io import read, write
from ase.constraints import FixAtoms

from GDPy.expedition.abstract import AbstractExpedition
from GDPy.reaction import create_reaction_explorer

from GDPy.utils.command import CustomTimer


class ReactionExplorer(AbstractExpedition):

    """ currently, use AFIR to search reaction pairs
    """

    name = "rxn" # exploration name

    creation_params = dict(
        opt_dname = "tmp_folder",
        struc_prefix = "cand"
    )

    collection_params = dict(
        traj_period = 1,
        selection_tags = ["TS", "FS", "optraj"]
    )

    def _parse_driver(self):
        """Create a driver with biased dynamics if needed."""

        return

    def _prior_create(self, input_params):
        """ some codes before creating exploratiosn of systems
            parse actions for this exploration from dict params
        """
        actions = super()._prior_create(input_params)

        # - get rxn searcher
        task_params = input_params["create"]["task"]
        rxn_search = create_reaction_explorer(task_params)
        rxn_search.logger = self.logger
        
        actions["reaction"] = rxn_search

        # - update driver params
        driver_params = input_params["create"]["driver"]
        # NOTE: we need a driver with biased dynamics functionality
        if "bias" not in driver_params:
            driver_params.update(bias=[])
        driver = self.pot_worker.potter.create_driver(driver_params)
        self.pot_worker.driver = driver

        actions["driver"] = driver

        return actions
    
    def _single_create(self, res_dpath, actions, data, *args, **kwargs):
        """"""
        # - generator
        frames = data["init_frames"]
        self.logger.info(f"number of initial structures: {len(frames)}")

        # - action
        # NOTE: create a separate calculation folder
        calc_dir_path = res_dpath / "create" / self.creation_params["opt_dname"]
        if not calc_dir_path.exists():
            calc_dir_path.mkdir(parents=True)
        #else:
        #    print(f"  {calc_dir_path.name} exists, so next...")
        rxn_search = actions["reaction"]
        worker = self.pot_worker

        is_finished = True
        for icand, atoms in enumerate(frames):
            self.logger.info(f"--- candidate {icand} ---")
            rxn_search.directory = calc_dir_path / (f"cand{icand}")
            rxn_search.run(worker, atoms)
        
        return is_finished
    
    def _single_collect(self, res_dpath, actions, data, *args, **kwargs):
        """"""
        # - generator
        frames = data["init_frames"]
        self.logger.info(f"number of initial structures: {len(frames)}")

        # - collect params
        traj_period = self.collection_params["traj_period"]

        create_dpath = res_dpath / "create"
        collect_dpath = res_dpath / "collect"

        calc_dir_path = create_dpath / self.creation_params["opt_dname"]

        rxn_search = actions["reaction"]

        # - harvest rxn results
        worker = self.pot_worker
        #   NOTE: we need pseudo pathways and some trajectories
        path_frames = []
        traj_frames = []
        for icand, atoms in enumerate(frames):
            rxn_search.directory = calc_dir_path / f"cand{icand}"
            ret = rxn_search.report(worker)
            #print(ret["pathways"], type(ret["pathways"]))
            # TODO: currently, structures are all mixed...
            for pathway in ret["pathways"]:
                    path_frames.extend(pathway)
            for traj_groups in ret["trajs"]:
                for traj in traj_groups:
                        traj_frames.extend(traj)
        write(collect_dpath/"path_frames.xyz", path_frames)
        write(collect_dpath/"traj_frames.xyz", path_frames)
        
        # - pass data
        data.update(
            **{
                "pot_frames_paths": path_frames,
                "pot_frames_trajs": traj_frames,
            }
        )

        return True


if __name__ == "__main__":
    pass