#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import copy
from typing import NoReturn, List

import numpy as np
import networkx as nx
from joblib import Parallel, delayed

from ase import Atoms
from ase.io import read, write

from .. import config
from .. import CustomTimer
from .. import StruGraphCreator, extract_chem_envs
from .. import paragroup_unique_chem_envs

from ..builder import StructureModifier
from ..group import create_a_group

DEFAULT_GRAPH_PARAMS = dict(
    pbc_grid = [2, 2, 0],
    graph_radius = 2,
    neigh_params = dict(
        covalent_ratio = 1.1,
        skin = 0.25
    )
)

def single_create_structure_graph(graph_params: dict, target_group: List[str], atoms: Atoms) -> List[nx.Graph]:
    """Create structure graph and get selected chemical environments.

    Find atoms with selected chemical symbols or in the defined region.

    Args:
        graph_params: Parameters for the graph representation.
        target_indices: A List of Integers.
        atoms: Input structure.

    Returns:
        A list of graphs that represent the chemical environments of selected atoms.

    """
    stru_creator = StruGraphCreator(**graph_params)

    natoms = len(atoms)
    group_indices = list(range(natoms))
    for command in target_group:
        curr_indices = create_a_group(atoms, command)
        group_indices = [i for i in group_indices if i in curr_indices]
    #config._print(f"{group_indices = }")

    graph = stru_creator.generate_graph(atoms, ads_indices=group_indices)

    graph_radius = stru_creator.graph_radius
    chem_envs = extract_chem_envs(graph, atoms, group_indices, graph_radius)
    #config._print(f"{graph_radius = }")

    return chem_envs


class GraphModifier(StructureModifier):

    
    def run(self, substrates: List[Atoms]=None, size: int=1, *args, **kwargs):
        """"""
        super().run(substrates=substrates, *args, **kwargs)

        prev_directory = self.directory
        curr_substrates = self.substrates

        for i in range(size):
            self.directory = prev_directory/f"graph-{i}"
            self.directory.mkdir(parents=True, exist_ok=True)
            cached_filepath = self.directory/"enumerated.xyz"
            if not cached_filepath.exists():
                self._print("-- run graph results --")
                modified_structures = self._irun(curr_substrates, )
                write(self.directory/"enumerated.xyz", modified_structures)
            else:
                self._print("-- use cached results --")
                modified_structures = read(self.directory/"enumerated.xyz", ":")
            n_structures = len(modified_structures)
            self._print(f"nframes: {n_structures}")
            curr_substrates = modified_structures
        
        self.directory = prev_directory

        return modified_structures 
    
    def _irun(substrates: List[Atoms]) -> List[Atoms]:
        """"""

        raise NotImplementedError()

    def _compare_structures(self, ret_frames: List[Atoms], graph_params, spec_params):
        """"""
        with CustomTimer(name="create-graphs", func=self._print):
            ret = Parallel(n_jobs=self.njobs)(
                delayed(single_create_structure_graph)(graph_params, spec_params, a)
                for a in ret_frames
            )
        # not unique across substrates
        #write(self.directory/f"possible_frames-{self.op_num}.xyz", ret_frames)

        # - check if the ret is empty
        #   it happens when all species are removed/exchanged...
        ret_envs = []
        for x in ret:
            ret_envs.extend(x)

        if ret_envs:
            ret_env_groups = ret
            self._print("Typical Chemical Environment "+str(ret_envs[0]))
            with CustomTimer(name="check-uniqueness", func=self._print):
                # compare chem envs
                #unique_envs, unique_groups = unique_chem_envs(
                #    chem_groups, list(enumerate(frames))
                #)
                unique_envs, unique_groups = paragroup_unique_chem_envs(
                    ret_env_groups, list(enumerate(ret_frames)), directory=self.directory, 
                    #n_jobs=self.njobs
                    n_jobs=1
                )
                #self._print("number of unique groups: ", len(unique_groups))

            # - get unique structures
            created_frames = [] # graphly unique
            for x in unique_groups:
                created_frames.append(x[0][1])
            ncandidates = len(created_frames)

            # -- unique info
            unique_data = []
            for i, x in enumerate(unique_groups):
                data = ["ug"+str(i)]
                data.extend([a[0] for a in x])
                unique_data.append(data)
            content = "# unique, indices\n"
            content += f"# ncandidates {ncandidates}\n"
            for d in unique_data:
                content += ("{:<8s}  "+"{:<8d}  "*(len(d)-1)+"\n").format(*d)

            unique_info_path = self.directory / f"unique-info.txt"
            with open(unique_info_path, "w") as fopen:
                fopen.write(content)
        else:
            self._print("Cant find valid species...")
            created_frames = ret_frames
            ncandidates = len(created_frames)

        return created_frames
    

if __name__ == "__main__":
    ...
