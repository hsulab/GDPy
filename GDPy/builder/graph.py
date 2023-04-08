#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
from typing import NoReturn, List

import numpy as np
import networkx as nx
from joblib import Parallel, delayed

from ase import Atoms
from ase.io import read, write

from GDPy import config
from GDPy.core.operation import Operation
from GDPy.builder.species import build_species
from GDPy.graph.creator import StruGraphCreator
from GDPy.graph.sites import SiteFinder
from GDPy.utils.command import CustomTimer
from GDPy.graph.comparison import get_unique_environments_based_on_bonds, paragroup_unique_chem_envs

DEFAULT_GRAPH_PARAMS = dict(
    pbc_grid = [2, 2, 0],
    graph_radius = 2,
    neigh_params = dict(
        covalent_ratio = 1.1,
        skin = 0.25
    )
)

def single_create_structure_graph(graph_params: dict, spec_params: dict, atoms: Atoms) -> List[nx.Graph]:
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

def single_insert_adsorbate(graph_params: dict, idx, atoms, ads, site_params: list, pfunc=print):
    """Insert adsorbate into the graph.
    """
    site_creator = SiteFinder(**graph_params)
    site_creator.pfunc = pfunc
    site_groups = site_creator.find(atoms, site_params)

    created_frames = []
    for i, (sites, params) in enumerate(zip(site_groups,site_params)):
        ads_params = params.get("ads", [{}])
        cur_frames = []
        for s in sites: 
            ads_frames = s.adsorb(
                ads, site_creator.ads_indices, ads_params
            )
            cur_frames.extend(ads_frames)
        created_frames.extend(cur_frames)
        pfunc(f"group {i} unique sites {len(sites)} with {len(cur_frames)} frames for substrate {idx}.")
    
    return created_frames

class insert_adsorbate_graph(Operation):

    def __init__(
            self, builder, species, adsorbate_elements: List[str], 
            sites: List[dict], graph: dict = DEFAULT_GRAPH_PARAMS
        ) -> NoReturn:
        """Insert an adsorbate on sites according to graph representation.
        """
        super().__init__([builder])

        self.species = species # make this a node

        self.adsorbate_elements = adsorbate_elements
        self.site_params = sites
        self.graph_params = graph

        #self.check_site_unique = True
        # adsorbate_indices
        # site_radius
        # region

        # - parallel
        self.njobs = config.NJOBS

        return
    
    def forward(self, substrates: List[Atoms]):
        """"""
        super().forward()

        cached_filepath = self.directory/"enumerated-last.xyz"
        if not cached_filepath.exists():
            ads_frames = self._insert_adsorbate(substrates, )
            write(self.directory/"enumerated-last.xyz", ads_frames)
        else:
            self.pfunc("Use cached results.")
            ads_frames = read(self.directory/"enumerated-last.xyz", ":")
        self.pfunc(f"nframes: {len(ads_frames)}")

        return ads_frames
    
    def _insert_adsorbate(self, substrates: List[Atoms]):
        """"""
        # -- parameters for finding adsorption sites
        site_params = copy.deepcopy(self.site_params)

        # -- parameters for site graph
        graph_params = copy.deepcopy(self.graph_params) # to create site graph
        adsorbate_elements = copy.deepcopy(self.adsorbate_elements)
        graph_params.update( 
            dict(
                adsorbate_elements = adsorbate_elements,
                #coordination_numbers = params.get("coordination_numbers"),
                site_radius = 2,
                #check_site_unique = params.get("check_site_unique", True)
            )
        )

        # -- parameters for species used for comparison
        species = self.species # to insert
        spec_params = dict(
            species = species,
            selected_species = [],
            spec_indices = None,
            region = None
        )

        # -------

        # -- 
        species = spec_params.get("species", None)
        self.pfunc(f"start to insert adsorbate {species}.")

        # - build adsorbate (single atom or molecule)
        #   and update selected_species
        if isinstance(species, str):
            # simple species
            adsorbate = build_species(species)
        else: # dict
            adsorbate = read(species["adsorbate"]) # only one structure
        symbols = list(set(adsorbate.get_chemical_symbols()))

        selected_species = copy.deepcopy(graph_params.get("adsorbate_elements", []))
        selected_species.extend(symbols)
        selected_species = list(set(selected_species))

        spec_params = copy.deepcopy(spec_params)
        spec_params["selected_species"] = selected_species

        # - get structures with inserted species
        with CustomTimer(name="insert-adsorbate", func=self.pfunc):
            #ret = Parallel(n_jobs=self.njobs)(
            ret = Parallel(n_jobs=1)(
                delayed(single_insert_adsorbate)(
                    graph_params, idx, a, adsorbate, site_params,
                    pfunc=self.pfunc
                ) for idx, a in enumerate(substrates)
            )
        ret_frames = []
        for frames in ret:
            ret_frames.extend(frames)
        write(self.directory/f"possible_frames.xyz", ret_frames)
        self.pfunc(f"nframes of inserted: {len(ret_frames)}")

        # NOTE: It is unnecessary to compare among substrates if the spectator
        #       adsorbates are not the same as the inserted one. Otherwise, 
        #       comparasion should be performed.
        created_frames = self._compare_structures(ret_frames, graph_params, spec_params)

        return created_frames

    def _compare_structures(self, ret_frames: List[Atoms], graph_params, spec_params):
        """"""
        with CustomTimer(name="create-graphs", func=self.pfunc):
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
            self.pfunc("Typical Chemical Environment "+str(ret_envs[0]))
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
                #self.pfunc("number of unique groups: ", len(unique_groups))

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
            self.pfunc("Cant find valid species...")
            created_frames = ret_frames
            ncandidates = len(created_frames)

        return created_frames


if __name__ == "__main__":
    ...