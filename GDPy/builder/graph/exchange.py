#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
from typing import List

from joblib import Parallel, delayed

from ase import Atoms
from ase.io import read, write

from GDPy import config
from GDPy.graph.creator import StruGraphCreator
from GDPy.graph.comparison import get_unique_environments_based_on_bonds, paragroup_unique_chem_envs
from GDPy.graph.utils import unpack_node_name
from GDPy.builder.graph.modifier import GraphModifier, DEFAULT_GRAPH_PARAMS
from GDPy.utils.command import CustomTimer


def single_exchange_adsorbate(graph_params: dict, spec_params: dict, atoms: Atoms):
    """Exchange selected particles from the structure with target species.

    Currently, only single atom can be removed. TODO: molecule.

    Args:
        graph_params: Parameters for creating graphs.
        spec_params: Parameters for finding species to remove.

    """
    # - create graph from structure
    stru_creator = StruGraphCreator(
        **graph_params
    )

    # - check if spec_indices are all species
    species = spec_params["species"]

    chemical_symbols = atoms.get_chemical_symbols()

    spec_indices = spec_params.get("spec_indices", None)
    if spec_indices is None:
        region = spec_params.get("region", None)
        if region is None:
            ads_indices = [] # selected indices
            for i, sym in enumerate(chemical_symbols):
                if sym == species:
                    ads_indices.append(i)
        else:
            ads_indices = []
            #print(region)
            (ox, oy, oz, xl, yl, zl, xh, yh, zh) = region
            for i, a in enumerate(atoms):
                if a.symbol == species:
                    pos = a.position
                    if (
                        (ox+xl <= pos[0] <= ox+xh) and
                        (oy+yl <= pos[1] <= oy+yh) and
                        (oz+zl <= pos[2] <= oz+zh)
                    ):
                        ads_indices.append(i)
    else:
        ads_indices = copy.deepcopy(spec_indices)
    #print("ads_indices: ", ads_indices)

    # TODO: tags for molecule?
    for i in ads_indices:
        if chemical_symbols[i] != species:
            raise RuntimeError("Species to remove is inconsistent for those by indices.")
    
    target_species = spec_params["target"] # TODO: a molecule?

    # NOTE: compare environments of host species
    # - get chem envs
    _ = stru_creator.generate_graph(atoms, ads_indices_=ads_indices)
    chem_envs = stru_creator.extract_chem_envs(atoms)
    #print("ex chem envs: ", len(chem_envs))
    #print(chem_envs[0])
    #print(chem_envs[1])
    #for i, g in enumerate(chem_envs):
    #    print(g)
    #    #plot_graph(g, f"graph-{i}.png")
    #    for (u, d) in g.nodes.data():
    #        if d["central_ads"]:
    #            print(u, d)
    # NOTE: for single atom adsorption,
    assert len(chem_envs) == len(ads_indices), "Single atoms group into one adsorbate. Try reducing the covalent radii."
    # TODO: for molecule adsorption

    # - find unique sites to remove for this structure
    unique_indices = get_unique_environments_based_on_bonds(chem_envs)
    unique_envs = [chem_envs[i] for i in unique_indices]

    # - create sctructures
    unique_frames = []
    for g in unique_envs:
        for (u, d) in g.nodes.data():
            if d["central_ads"]:
                chem_sym, idx, offset = unpack_node_name(u)
                if chem_sym == species:
                    #new_atoms = make_clean_atoms(atoms)
                    new_atoms = atoms.copy()
                    new_atoms[idx].symbol = target_species # TODO: a molecule?
                    unique_frames.append(new_atoms)
                    break
        else:
            # no valid adsorbate for this structure
            ...

    return unique_frames, unique_envs


class GraphExchangeModifier(GraphModifier):

    def __init__(
            self, species: str, target: str, target_indices: List[int], adsorbate_elements: List[str], 
            substrates=None, graph: dict = DEFAULT_GRAPH_PARAMS,
            *args, **kwargs
        ):
        """Insert an adsorbate on sites according to graph representation.
        """
        super().__init__(substrates=substrates, *args, **kwargs)

        self.species = species # make this a node
        self.target = target

        self.target_indices = target_indices

        self.adsorbate_elements = adsorbate_elements
        self.graph_params = graph

        #self.check_site_unique = True
        # adsorbate_indices
        # site_radius
        # region

        # - parallel
        self.njobs = config.NJOBS

        return
    
    def _irun(self, substrates: List[Atoms]) -> List[Atoms]:
        """Exchange an adsorbate with another species.
        """
        self._print("---run exchange---")
        # - params for graph creator
        graph_params = copy.deepcopy(self.graph_params)
        adsorbate_elements = copy.deepcopy(self.adsorbate_elements)
        graph_params.update(adsorbate_elements=adsorbate_elements)

        # - find species to exchange
        species = self.species # host species
        target = self.target # parasite species
        spec_params = dict(
            species = species,
            target = target,
            selected_species = [species, target],
            spec_indices = self.target_indices,
            region = None
        )

        # - get possible sites to exchange
        with CustomTimer(name="exchange-adsorbate", func=self._print):
            ret = Parallel(n_jobs=self.njobs)(
                delayed(single_exchange_adsorbate)(graph_params, spec_params, a) 
                for a in substrates
            )

            ret_frames, ret_envs = [], []
            for i, (frames, envs) in enumerate(ret):
                nenvs = len(envs)
                # TODO: add info since it may be lost in atoms.copy() function
                #for a in frames:
                #    a.info["subid"] = subid
                # -- add data
                ret_envs.extend(envs)
                ret_frames.extend(frames)
                self._print(f"number of sites {nenvs} to exchange for substrate {i}.")
        # not unique across substrates
        write(self.directory/f"possible_frames.xyz", ret_frames)

        # - compare the graph of chemical environments in the structure
        #   NOTE: if Zn atoms were to exchange with Cr, the chem envs of 
        #         the rest Zn atoms are used to compare the structure difference.
        #         TODO: consider Cr as well?
        created_frames = self._compare_structures(ret_frames, graph_params, spec_params)

        return created_frames



if __name__ == "__main__":
    ...