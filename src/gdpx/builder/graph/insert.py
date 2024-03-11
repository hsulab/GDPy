#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
from typing import NoReturn, Callable, List

import numpy as np
import networkx as nx
from joblib import Parallel, delayed

import ase
from ase import Atoms
from ase.io import read, write


from .. import config
from .. import CustomTimer
from .. import SiteFinder
from .modifier import GraphModifier, DEFAULT_GRAPH_PARAMS


def str2atoms(species: str) -> Atoms:
    """Convert a string to an Atoms object."""
    # - build adsorbate
    atoms = None
    if species in ase.data.chemical_symbols:
        atoms = Atoms(species, positions=[[0.0, 0.0, 0.0]])
    elif species in ase.collections.g2.names:
        atoms = ase.build.molecule(species)
    else:
        raise ValueError(f"Fail to create species {species}")

    return atoms


def single_insert_adsorbate(
    graph_params: dict, idx, atoms, ads, site_params: list, 
    print_func: Callable=print, debug_func: Callable=print
):
    """Insert adsorbate into the graph.
    """
    site_creator = SiteFinder(**graph_params)
    site_creator._print = print_func
    site_creator._debug = debug_func
    site_groups = site_creator.find(atoms, site_params)

    ads_indices = [a.index for a in atoms if a.symbol in site_creator.adsorbate_elements]

    created_frames = []
    for i, (sites, params) in enumerate(zip(site_groups, site_params)):
        ads_params = params.get("ads", [{}])
        cur_frames = []
        for s in sites: 
            ads_frames = s.adsorb(
                ads, ads_indices, ads_params
            )
            cur_frames.extend(ads_frames)
        created_frames.extend(cur_frames)
        print_func(
            f"group {i} unique sites {len(sites)} with {len(cur_frames)} frames for substrate {idx}."
        )
    
    return created_frames


class GraphInsertModifier(GraphModifier):

    name = "graph_insert"

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
            adsorbate = str2atoms(species)
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
                    print_func=self._print, debug_func=self._debug
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
        created_frames = self._compare_structures(
            ret_frames, graph_params, target_group
        )

        return created_frames


if __name__ == "__main__":
    ...
