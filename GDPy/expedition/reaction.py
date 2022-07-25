#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import NoReturn
from unittest.result import failfast

from ase.io import read, write
from ase.constraints import FixAtoms

from GDPy.expedition.abstract import AbstractExplorer
from GDPy.reaction.AFIR import AFIRSearch

from GDPy.builder.constraints import parse_constraint_info


class ReactionExplorer(AbstractExplorer):

    """ currently, use AFIR to search reaction pairs
    """


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
                    pass

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
                for icand, atoms in enumerate(frames):
                    # --- TODO: check constraints on atoms
                    #           actually this should be in a dynamics object
                    mobile_indices, frozen_indices = parse_constraint_info(atoms, sys_cons_text, ret_text=False)
                    if frozen_indices:
                        atoms.set_constraint(FixAtoms(indices=frozen_indices))

                    print(f"--- candidate {icand} ---")
                    afir_search.directory = res_dir / (f"cand{icand}")
                    afir_search.run(atoms, calc)
                    #break

        return

    def icollect(self, exp_name, working_directory):
        """
        """
        pass


if __name__ == "__main__":
    pass