#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
from typing import Callable, List, Tuple

import networkx as nx
from ase import Atoms
from ase.io import write
from joblib import Parallel, delayed

from gdpx.group.group import create_a_group

from .. import (
    CustomTimer,
    StruGraphCreator,
    extract_chem_envs,
    get_unique_environments_based_on_bonds,
    unpack_node_name,
)
from .modifier import DEFAULT_GRAPH_PARAMS, GraphModifier


def single_remove_adsorbate(
    species: str,
    graph_params: dict,
    target_group: List[dict],
    atoms: Atoms,
    print_func: Callable = print,
    debug_func: Callable = print,
) -> Tuple[List[Atoms], List[nx.Graph]]:
    """Remove selected particles from the structure.

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
        if chemical_symbols[i] != species:
            raise RuntimeError(
                "Species to remove is inconsistent for those by indices."
            )

    # - get chem envs
    graph = stru_creator.generate_graph(atoms, ads_indices=group_indices)
    chem_envs = extract_chem_envs(
        graph, atoms, group_indices, stru_creator.graph_radius
    )

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
                    new_atoms = atoms.copy()
                    del new_atoms[idx]
                    unique_frames.append(new_atoms)
                    break
        else:
            # no valid adsorbate for this structure
            ...

    return unique_frames, unique_envs


class GraphRemoveModifier(GraphModifier):

    name: str = "graph_remove"

    def __init__(
        self,
        species,
        spectators: List[str],
        target_group: List[dict],
        substrates=None,
        graph: dict = DEFAULT_GRAPH_PARAMS,
        *args,
        **kwargs,
    ):
        """Insert an adsorbate on sites according to graph representation."""
        super().__init__(substrates=substrates, *args, **kwargs)

        self.species = species  # make this a node

        self.target_group = target_group

        self.spectators = spectators
        self.graph_params = graph

        # self.check_site_unique = True
        # adsorbate_indices
        # site_radius
        # region

        return

    def _irun(self, substrates: List[Atoms]) -> List[Atoms]:
        """Remove atoms/molecules/adsorbates."""
        self._print("---run remove---")
        graph_params = copy.deepcopy(self.graph_params)
        adsorbate_elements = copy.deepcopy(self.spectators)
        graph_params.update(adsorbate_elements=adsorbate_elements)

        # - get chem envs of selected species that may be removed
        with CustomTimer(name="remove-adsorbate", func=self._print):
            ret = Parallel(n_jobs=self.njobs)(
                delayed(single_remove_adsorbate)(
                    self.species,
                    graph_params,
                    self.target_group,
                    a,
                    print_func=self._print,
                    debug_func=self._debug,
                )
                for idx, a in enumerate(substrates)
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
                self._print(f"number of sites {nenvs} to remove for substrate {i}.")
        # nsites = len(ret_frames)
        # self._print(f"Total number of chemical environments: {nsites}")

        # - further unique envs among different substrates
        #   only compare chemical environments
        # unique_indices = get_unique_environments_based_on_bonds(ret_envs)
        # created_frames = [ret_frames[i] for i in unique_indices]

        # - compare the graph of chemical environments in the structure
        #   NOTE: if O atoms were to remove, the chem envs of the rest O atoms
        #         are used to compare the structure difference.
        write(self.directory / f"possible_frames.xyz", ret_frames)

        # - get unique structures among substrates
        created_frames = self._compare_structures(
            ret_frames, graph_params, self.target_group
        )

        return created_frames


if __name__ == "__main__":
    ...
