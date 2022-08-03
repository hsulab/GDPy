#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

from typing import NoReturn
from unittest.result import failfast

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
        calc_dir_name = "tmp_folder"
    )

    collection_params = dict(
        resdir_name = "sorted",
        selection_tags = ["TS", "FS", "optraj"]
    )


    def icreate(self, exp_name, working_directory) -> NoReturn:
        """ create and submit exploration tasks
        """
        # - a few info
        exp_dict = self.explorations[exp_name]
        included_systems = exp_dict.get("systems", None)

        # - create action
        afir_params = exp_dict["creation"]["AFIR"]
        afir_search = AFIRSearch(**afir_params)
        
        calc = self.pot_manager.calc

        # - run systems
        if included_systems is not None:
            for slabel in included_systems:
                print(f"----- Explore System {slabel} -----")

                # - prepare output directory
                res_dir = working_directory / exp_name / slabel
                if not res_dir.exists():
                    res_dir.mkdir(parents=True)
                else:
                    print(f"  {res_dir.name} exists, so next...")
                    continue

                # - read substrate
                system_dict = self.init_systems.get(slabel, None) # system name
                if system_dict is None:
                    raise ValueError(f"Find unexpected system {system_dict}.")
                sys_cons_text = system_dict.get("constraint", None)
                
                # - read start frames
                # the expedition can start with different initial configurations
                stru_path = system_dict["structure"]
                frames = read(stru_path, ":")

                print("number of frames: ", len(frames))
                
                # - action
                # NOTE: create a separate calculation folder
                calc_dir_path = res_dir / self.creation_params["calc_dir_name"]
                if not calc_dir_path.exists():
                    calc_dir_path.mkdir(parents=True)
                else:
                    print(f"  {calc_dir_path.name} exists, so next...")
                    continue

                for icand, atoms in enumerate(frames):
                    # --- TODO: check constraints on atoms
                    #           actually this should be in a dynamics object
                    mobile_indices, frozen_indices = parse_constraint_info(atoms, sys_cons_text, ret_text=False)
                    if frozen_indices:
                        atoms.set_constraint(FixAtoms(indices=frozen_indices))

                    print(f"--- candidate {icand} ---")
                    afir_search.directory = calc_dir_path / (f"cand{icand}")
                    afir_search.run(atoms, calc)
                    #break

        return

    def icollect(self, exp_name, working_directory) -> NoReturn:
        """
        """
        # - a few info
        exp_dict = self.explorations[exp_name]
        included_systems = exp_dict.get("systems", None)

        # - collect action TODO: move this to main?
        collection_params = exp_dict["collection"]
        traj_period = collection_params.get("traj_period", 1)
        print(f"traj_period: {traj_period}")

        # - run systems
        if included_systems is not None:
            for slabel in included_systems:
                print(f"----- Explore System {slabel} -----")

                # - prepare output directory
                res_dir = working_directory / exp_name / slabel
                if not res_dir.exists():
                    # res_dir.mkdir(parents=True)
                    print(f"  {res_dir.name} does not exist, so next...")
                    continue
                else:
                    pass

                calc_dir_path = res_dir / self.creation_params["calc_dir_name"]

                sorted_dir = working_directory / exp_name / slabel / "sorted"
                if not sorted_dir.exists():
                    sorted_dir.mkdir(parents=True)
                else:
                    print(f"  {sorted_dir.name} does not exist, so next...")
                    continue

                # - find opt-trajs and pseudo_pathway
                #   cand0/f0/g0 and pseudo_pathway.xyz
                # NOTE: what we need
                #       approx. TS and related trajs
                approx_TSs = []
                approx_FSs = []
                optraj_frames = []

                with CustomTimer("read-structure"):
                    # TODO: check output exists?
                    if (sorted_dir/"optraj_frames.xyz").exists():
                        approx_TSs = read(sorted_dir/"approx_TSs.xyz", ":")
                        approx_FSs = read(sorted_dir/"approx_FSs.xyz", ":")
                        optraj_frames = read(sorted_dir/"optraj_frames.xyz", ":")
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
                                    a.info["comment"] = "-".join([cand_dir.name, reac_dir.name, "pseudO_path", str(i)])
                                approx_FSs.append(path_frames[-1])

                                energies = [a.get_potential_energy() for a in path_frames]
                                max_idx = np.argmax(energies)
                                approx_TSs.append(path_frames[max_idx])
                
                        write(sorted_dir/"approx_TSs.xyz", approx_TSs)
                        write(sorted_dir/"approx_FSs.xyz", approx_FSs)
                        write(sorted_dir/"optraj_frames.xyz", optraj_frames)

                # - selection
                # 1. select approx. TS and FS
                # 2. select opt trajs
                selection_params = exp_dict.get("selection", None)
                selector = create_selector(selection_params, directory=sorted_dir)

                if selector:
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