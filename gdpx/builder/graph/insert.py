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
from gdpx.builder.species import build_species
from gdpx.graph.creator import StruGraphCreator
from gdpx.graph.sites import SiteFinder
from gdpx.utils.command import CustomTimer
from gdpx.graph.comparison import get_unique_environments_based_on_bonds, paragroup_unique_chem_envs

from gdpx.builder.graph.modifier import GraphModifier, DEFAULT_GRAPH_PARAMS


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


class GraphInsertModifier(GraphModifier):

    def __init__(
            self, species, spectators: List[str], 
            sites: List[dict], substrates=None, graph: dict = DEFAULT_GRAPH_PARAMS,
            *args, **kwargs
        ):
        """Insert an adsorbate on sites according to graph representation.
        """
        super().__init__(substrates=substrates, *args, **kwargs)

        self.species = species # make this a node

        self.spectators = spectators 
        self.site_params = sites
        self.graph_params = graph

        #self.check_site_unique = True
        # adsorbate_indices
        # site_radius
        # region

        # - parallel
        self.njobs = config.NJOBS

        return
    
    def _irun(self, substrates: List[Atoms]) -> List[Atoms]:
        """Insert an adsorabte on the substrate."""
        # -- parameters for finding adsorption sites
        site_params = copy.deepcopy(self.site_params)

        # -- parameters for site graph
        graph_params = copy.deepcopy(self.graph_params) # to create site graph
        adsorbate_elements = copy.deepcopy(self.spectators)
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
        self._print(f"start to insert adsorbate {species}.")

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

        # - get structures with inserted species
        with CustomTimer(name="insert-adsorbate", func=self._print):
            #ret = Parallel(n_jobs=self.njobs)(
            ret = Parallel(n_jobs=1)(
                delayed(single_insert_adsorbate)(
                    graph_params, idx, a, adsorbate, site_params,
                    pfunc=self._print
                ) for idx, a in enumerate(substrates)
            )
        ret_frames = []
        for frames in ret:
            ret_frames.extend(frames)
        write(self.directory/f"possible_frames.xyz", ret_frames)
        self._print(f"nframes of inserted: {len(ret_frames)}")

        # NOTE: It is unnecessary to compare among substrates if the spectator
        #       adsorbates are not the same as the inserted one. Otherwise, 
        #       comparasion should be performed.
        target_group = ["symbol "+" ".join(selected_species)]
        created_frames = self._compare_structures(ret_frames, graph_params, target_group)

        return created_frames
    

if __name__ == "__main__":
    ...