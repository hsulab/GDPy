#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
from typing import Callable, Tuple, List

import networkx as nx

from joblib import Parallel, delayed

from ase import Atoms
from ase.io import read, write

from .. import CustomTimer
from .. import StruGraphCreator, extract_chem_envs
from .. import get_unique_environments_based_on_bonds
from .. import unpack_node_name

from .modifier import GraphModifier, DEFAULT_GRAPH_PARAMS
from ..group import create_a_group


def single_exchange_adsorbate(
    graph_params: dict,
    species: str,
    target,
    target_group,
    atoms: Atoms,
    print_func: Callable = print,
    debug_func: Callable = print,
) -> Tuple[List[Atoms], List[nx.Graph]]:
    """Exchange selected particles from the structure with target species.

    Currently, only single atom can be removed. TODO: molecule.

    Args:
        graph_params: Parameters for creating graphs.
        spec_params: Parameters for finding species to remove.

    """
    # - create graph from structure
    stru_creator = StruGraphCreator(**graph_params)

    # - check if spec_indices are all species
    natoms = len(atoms)
    group_indices = list(range(natoms))
    for command in target_group:
        curr_indices = create_a_group(atoms, command)
        group_indices = [i for i in group_indices if i in curr_indices]
    debug_func(f"group_indices to remove {group_indices}")

    # TODO: tags for molecule?
    chemical_symbols = atoms.get_chemical_symbols()
    for i in group_indices:
        if chemical_symbols[i] == species:
            break
    else:
        raise RuntimeError(f"There is no {species} to remove by target group.")

    # NOTE: compare environments of host species
    # - get chem envs
    graph = stru_creator.generate_graph(atoms, ads_indices=group_indices)
    chem_envs = extract_chem_envs(
        graph, atoms, group_indices, stru_creator.graph_radius
    )
    # print("ex chem envs: ", len(chem_envs))
    # print(chem_envs[0])
    # print(chem_envs[1])
    # for i, g in enumerate(chem_envs):
    #    print(g)
    #    #plot_graph(g, f"graph-{i}.png")
    #    for (u, d) in g.nodes.data():
    #        if d["central_ads"]:
    #            print(u, d)
    # NOTE: for single atom adsorption,
    assert len(chem_envs) == len(
        group_indices
    ), "Single atoms group into one adsorbate. Try reducing the covalent radii."
    # TODO: for molecule adsorption

    # - find unique sites to remove for this structure
    unique_indices = get_unique_environments_based_on_bonds(chem_envs)
    unique_envs = [chem_envs[i] for i in unique_indices]

    # - create sctructures
    unique_frames = []
    for g in unique_envs:
        for u, d in g.nodes.data():
            if d["central_ads"]:
                chem_sym, idx, offset = unpack_node_name(u)
                if chem_sym == species:
                    # new_atoms = make_clean_atoms(atoms)
                    new_atoms = atoms.copy()
                    new_atoms[idx].symbol = target  # TODO: a molecule?
                    unique_frames.append(new_atoms)
                    break
        else:
            # no valid adsorbate for this structure
            ...

    return unique_frames, unique_envs


class GraphExchangeModifier(GraphModifier):

    def __init__(
        self,
        species: str,
        target: str,
        target_group,
        spectators: List[str],
        substrates=None,
        graph: dict = DEFAULT_GRAPH_PARAMS,
        *args,
        **kwargs,
    ):
        """Insert an adsorbate on sites according to graph representation."""
        super().__init__(substrates=substrates, *args, **kwargs)

        self.species = species  # make this a node
        self.target = target

        self.target_group = target_group

        self.spectators = spectators
        self.graph_params = graph

        # self.check_site_unique = True
        # adsorbate_indices
        # site_radius
        # region

        return

    def _irun(self, substrates: List[Atoms]) -> List[Atoms]:
        """Exchange an adsorbate with another species."""
        self._print("---run exchange---")
        # - params for graph creator
        graph_params = copy.deepcopy(self.graph_params)
        adsorbate_elements = copy.deepcopy(self.spectators)
        graph_params.update(adsorbate_elements=adsorbate_elements)

        # - get possible sites to exchange
        with CustomTimer(name="exchange-adsorbate", func=self._print):
            ret = Parallel(n_jobs=self.njobs)(
                delayed(single_exchange_adsorbate)(
                    graph_params,
                    self.species,
                    self.target,
                    self.target_group,
                    a,
                    print_func=self._print,
                    debug_func=self._debug,
                )
                for a in substrates
            )

            ret_frames, ret_envs = [], []
            for i, (frames, envs) in enumerate(ret):
                nenvs = len(envs)
                # TODO: add info since it may be lost in atoms.copy() function
                # for a in frames:
                #    a.info["subid"] = subid
                # -- add data
                ret_envs.extend(envs)
                ret_frames.extend(frames)
                self._print(f"number of sites {nenvs} to exchange for substrate {i}.")
        # not unique across substrates
        write(self.directory / f"possible_frames.xyz", ret_frames)

        # - compare the graph of chemical environments in the structure
        #   NOTE: if Zn atoms were to exchange with Cr, the chem envs of
        #         the rest Zn atoms are used to compare the structure difference.
        #         TODO: consider Cr as well?
        created_frames = self._compare_structures(
            ret_frames, graph_params, self.target_group
        )

        return created_frames


if __name__ == "__main__":
    ...
