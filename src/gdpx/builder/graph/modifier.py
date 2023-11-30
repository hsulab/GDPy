#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import copy
from typing import NoReturn, List

import numpy as np
import networkx as nx
from joblib import Parallel, delayed

from ase import Atoms
from ase.io import read, write

from gdpx import config
from gdpx.core.register import registers
from gdpx.builder.species import build_species
from gdpx.graph.creator import StruGraphCreator
from gdpx.graph.sites import SiteFinder
from gdpx.utils.command import CustomTimer
from gdpx.graph.comparison import get_unique_environments_based_on_bonds, paragroup_unique_chem_envs

from gdpx.builder.builder import StructureModifier
from ..group import create_a_group

DEFAULT_GRAPH_PARAMS = dict(
    pbc_grid = [2, 2, 0],
    graph_radius = 2,
    neigh_params = dict(
        covalent_ratio = 1.1,
        skin = 0.25
    )
)

def single_create_structure_graph(graph_params: dict, target_group, atoms: Atoms) -> List[nx.Graph]:
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

    config._debug(f"target_group: {target_group}")

    natoms = len(atoms)
    group_indices = list(range(natoms))
    for command in target_group:
        curr_indices = create_a_group(atoms, command)
        group_indices = [i for i in group_indices if i in curr_indices]

    _ = stru_creator.generate_graph(atoms, ads_indices_=group_indices)

    chem_envs = stru_creator.extract_chem_envs(atoms)

    return chem_envs

def _temp_single_create_structure_graph(graph_params: dict, spec_params: dict, atoms: Atoms) -> List[nx.Graph]:
    """Create structure graph and get selected chemical environments.

    Find atoms with selected chemical symbols or in the defined region.

    Args:
        graph_params: Parameters for the graph representation.
        spec_params: Selected criteria.
        atoms: Input structure.

    Returns:
        A list of graphs that represent the chemical environments of selected atoms.

    """
    stru_creator = StruGraphCreator(**graph_params)

    # - check if spec_indices are all species
    selected_species = spec_params["selected_species"]
    #print(selected_species)

    chemical_symbols = atoms.get_chemical_symbols()
    #print(chemical_symbols)

    spec_indices = spec_params.get("spec_indices", None)
    if spec_indices is None:
        region = spec_params.get("region", None)
        if region is None:
            ads_indices = [] # selected indices
            for i, sym in enumerate(chemical_symbols):
                if sym in selected_species:
                    ads_indices.append(i)
        else:
            ads_indices = []
            #print(region)
            (ox, oy, oz, xl, yl, zl, xh, yh, zh) = region
            for i, a in enumerate(atoms):
                if a.symbol in selected_species:
                    pos = a.position
                    if (
                        (ox+xl <= pos[0] <= ox+xh) and
                        (oy+yl <= pos[1] <= oy+yh) and
                        (oz+zl <= pos[2] <= oz+zh)
                    ):
                        ads_indices.append(i)
    else:
        ads_indices = copy.deepcopy(spec_indices)
    #print(ads_indices)

    _ = stru_creator.generate_graph(atoms, ads_indices_=ads_indices)

    #print(stru_creator.graph)
    #return stru_creator.graph

    chem_envs = stru_creator.extract_chem_envs(atoms)
    return chem_envs


class GraphModifier(StructureModifier):

    
    def run(self, substrates: List[Atoms]=None, size: int=1, *args, **kwargs):
        """"""
        super().run(substrates=substrates, *args, **kwargs)

        cached_filepath = self.directory/"enumerated.xyz"
        if not cached_filepath.exists():
            modified_structures = self._irun(self.substrates, )
            write(self.directory/"enumerated.xyz", modified_structures)
        else:
            self._print("Use cached results.")
            modified_structures = read(self.directory/"enumerated.xyz", ":")
        n_structures = len(modified_structures)
        self._print(f"nframes: {n_structures}")

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
            with CustomTimer(name="check-uniqueness"):
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
