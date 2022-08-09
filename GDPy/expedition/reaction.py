#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

from typing import NoReturn

from ase.io import read, write
from ase.constraints import FixAtoms

from GDPy.expedition.abstract import AbstractExplorer
from GDPy.reaction.AFIR import AFIRSearch

from GDPy.builder.constraints import parse_constraint_info
from GDPy.selector.abstract import create_selector

from GDPy.utils.command import CustomTimer


class ReactionExplorer(AbstractExplorer):

    """ currently, use AFIR to search reaction pairs
    """

    name = "rxn" # exploration name

    creation_params = dict(
        opt_dname = "tmp_folder",
        struc_prefix = "cand"
    )

    collection_params = dict(
        resdir_name = "sorted",
        selection_tags = ["TS", "FS", "optraj"]
    )

    def _prior_create(self, input_params):
        """ some codes before creating exploratiosn of systems
            parse actions for this exploration from dict params
        """
        selector = super()._prior_create(input_params)

        # NOTE: currently, we only have AFIR...
        afir_params = input_params["create"]["AFIR"]
        afir_search = AFIRSearch(**afir_params)
        
        calc = self.pot_manager.calc

        actions = {}
        actions["reaction"] = afir_search

        return actions, selector
    
    def _single_create(self, res_dpath, frames, cons_text, actions, *args, **kwargs):
        """"""
        super()._single_create(res_dpath, frames, cons_text, actions, *args, **kwargs)

        # - action
        # NOTE: create a separate calculation folder
        calc_dir_path = res_dpath / "create" / self.creation_params["opt_dname"]
        if not calc_dir_path.exists():
            calc_dir_path.mkdir(parents=True)
        #else:
        #    print(f"  {calc_dir_path.name} exists, so next...")

        for icand, atoms in enumerate(frames):
            # --- TODO: check constraints on atoms
            #           actually this should be in a dynamics object
            mobile_indices, frozen_indices = parse_constraint_info(atoms, cons_text, ret_text=False)
            if frozen_indices:
                atoms.set_constraint(FixAtoms(indices=frozen_indices))

            print(f"--- candidate {icand} ---")
            actions["reaction"].directory = calc_dir_path / (f"cand{icand}")
            actions["reaction"].run(atoms, self.pot_manager.calc)
            #break
        
        return
    
    def _single_collect(self, res_dpath, frames, cons_text, actions, selector, *args, **kwargs):
        """"""
        super()._single_collect(res_dpath, frames, cons_text, actions, *args, **kwargs)

        traj_period = self.creation_params["traj_period"]

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

        with CustomTimer("read-structure"):
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

        # - select
        sorted_dir = res_dpath / "sorted"

        if selector:
            if not sorted_dir.exists():
                sorted_dir.mkdir(parents=True)
            else:
                #print(f"  {sorted_dir.name} does not exist, so next...")
                pass
                
            selector.directory = sorted_dir

            candidate_group = dict(
                TS = approx_TSs,
                FS = approx_FSs,
                optraj = optraj_frames
            )
            for prefix, cur_frames in candidate_group.items():
                # TODO: add info to selected frames
                # TODO: select based on minima (Trajectory-based Boltzmann)
                print(f"--- Selection Method {selector.name} for {prefix} ---")
                #print("ncandidates: ", len(cur_frames))
                # NOTE: there is an index map between traj_indices and selected_indices
                selector.prefix = prefix
                cur_frames = selector.select(cur_frames)
                #print("nselected: ", len(cur_frames))
                #write(sorted_path/f"{selector.name}-selected-{isele}.xyz", cur_frames)

        return


if __name__ == "__main__":
    pass