#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

from typing import NoReturn

from ase.io import read, write
from ase.constraints import FixAtoms

from GDPy.expedition.abstract import AbstractExpedition
from GDPy.reaction.AFIR import AFIRSearch

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

    def _prior_create(self, input_params):
        """ some codes before creating exploratiosn of systems
            parse actions for this exploration from dict params
        """
        actions = super()._prior_create(input_params)

        # NOTE: currently, we only have AFIR...
        afir_params = input_params["create"]["AFIR"]
        afir_search = AFIRSearch(**afir_params)
        
        actions["reaction"] = afir_search

        return actions
    
    def _single_create(self, res_dpath, actions, data, *args, **kwargs):
        """"""
        # - generator
        generator = actions["generator"]
        self.logger.info(generator.__class__.__name__)
        frames = generator.run(kwargs.get("ran_size", 1))
        self.logger.info(f"number of initial structures: {len(frames)}")
        from GDPy.builder.direct import DirectGenerator
        actions["generator"] = DirectGenerator(frames, res_dpath/"init")

        # - action
        # NOTE: create a separate calculation folder
        calc_dir_path = res_dpath / "create" / self.creation_params["opt_dname"]
        if not calc_dir_path.exists():
            calc_dir_path.mkdir(parents=True)
        #else:
        #    print(f"  {calc_dir_path.name} exists, so next...")

        is_finished = True
        for icand, atoms in enumerate(frames):
            self.logger.info(f"--- candidate {icand} ---")
            actions["reaction"].directory = calc_dir_path / (f"cand{icand}")
            actions["reaction"].run(atoms, self.pot_worker.potter.calc)
        
        return is_finished
    
    def _single_collect(self, res_dpath, actions, data, *args, **kwargs):
        """"""
        traj_period = self.collection_params["traj_period"]

        create_dpath = res_dpath / "create"
        collect_dpath = res_dpath / "collect"

        calc_dir_path = create_dpath / self.creation_params["opt_dname"]

        # - find opt-trajs and pseudo_pathway
        #   cand0/f0/g0 and pseudo_pathway.xyz
        # NOTE: what we need
        #       approx. TS and related trajs
        approx_TSs = []
        approx_FSs = []
        optraj_frames = []

        with CustomTimer("read-structure", func=self.logger.info):
            # TODO: check output exists?
            if (collect_dpath/"optraj_frames.xyz").exists():
                approx_TSs = read(collect_dpath/"approx_TSs.xyz", ":")
                approx_FSs = read(collect_dpath/"approx_FSs.xyz", ":")
                optraj_frames = read(collect_dpath/"optraj_frames.xyz", ":")
            else:
                cand_dirs = calc_dir_path.glob("cand*") # TODO: user-defined dir prefix
                cand_dirs = sorted(cand_dirs, key=lambda x: int(x.name.strip("cand")))
                #print(cand_dirs)
                # TODO: joblib?
                for cand_dir in cand_dirs:
                    # --- find reactions
                    reac_dirs = cand_dir.glob("f*")
                    reac_dirs = sorted(reac_dirs, key=lambda x: int(x.name.strip("f")))
                    for reac_dir in reac_dirs:
                        # ----- find biased opt trajs
                        # TODO: make this a utility function?
                        gam_dirs = reac_dir.glob("g*")
                        gam_dirs = sorted(gam_dirs, key=lambda x: int(x.name.strip("g")))

                        # optrajs
                        for gam_dir in gam_dirs:
                            traj_frames = read(gam_dir/"traj.xyz", ":")
                            for i, a in enumerate(traj_frames):
                                a.info["comment"] = "-".join([cand_dir.name, reac_dir.name, gam_dir.name, str(i)])
                            traj_indices = list(range(1,len(traj_frames)-1,traj_period))
                            optraj_frames.extend([traj_frames[ti] for ti in traj_indices]) # NOTE: first is IS, last is in pseudo
                        # pathway, find TS and FS
                        path_frames = read(reac_dir/"pseudo_path.xyz", ":")
                        for i, a in enumerate(path_frames):
                            a.info["comment"] = "-".join([cand_dir.name, reac_dir.name, "pseudo_path", str(i)])
                        approx_FSs.append(path_frames[-1])

                        energies = [a.get_potential_energy() for a in path_frames]
                        max_idx = np.argmax(energies)
                        approx_TSs.append(path_frames[max_idx])
                
                write(collect_dpath/"approx_TSs.xyz", approx_TSs)
                write(collect_dpath/"approx_FSs.xyz", approx_FSs)
                write(collect_dpath/"optraj_frames.xyz", optraj_frames)
        
        # - pass data
        data.update(
            **{
                "pot_frames_TS": approx_TSs,
                "pot_frames_FS": approx_FSs,
                "pot_frames_optraj": optraj_frames,
            }
        )

        return True


if __name__ == "__main__":
    pass