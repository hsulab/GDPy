#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import copy
import itertools
from typing import Optional

import networkx as nx
import numpy as np
import numpy.typing
from ase import Atoms
from ase.formula import Formula
from ase.geometry import find_mic
from ase.neighborlist import NeighborList, natural_cutoffs

from gdpx.group import evaluate_group_expression

from .describer import BaseDescriber


def reassemble_fragment_by_depth_first_search(atoms: Atoms, neighlist: NeighborList, start_index: int) -> None:
    """Ensure that atoms in the fragment have proper connectivities.

    This uses a neighbour list with bothways=True.

    Note:
        This changes the positions of the input atoms object.

    Args:
        atoms: Atoms object.
        neighlist: NeighborList object.
        start_index: The index of the atom to start the reassembly.

    """
    cell = atoms.cell.array

    def dfs_stack(
        cell: numpy.typing.NDArray,
        prev_positions: numpy.typing.NDArray,
        positions: numpy.typing.NDArray,
        start_index: int,
    ):
        """"""
        visited = set()

        stack = [(start_index, positions[start_index])]
        while stack:
            index, ref_pos = stack.pop()
            if index in visited:
                continue
            visited.add(index)

            indices, offsets = neighlist.get_neighbors(index)
            for j, offset in zip(indices, offsets):
                if j not in visited:
                    # Calculate the distance vector considering periodic boundary conditions
                    dist_vec = prev_positions[j] + np.dot(offset, cell) - ref_pos
                    dist_vec -= np.dot(np.round(dist_vec / cell.diagonal()), cell)
                    positions[j] = ref_pos + dist_vec
                    stack.append((j, positions[j]))

        return positions

    positions = dfs_stack(cell, atoms.positions, copy.deepcopy(atoms.positions), start_index)

    atoms.positions = positions

    return


def create_a_graph(atoms: Atoms, neighlist: NeighborList, indices: Optional[list[int]] = None) -> nx.Graph:
    """Create a graph.

    This uses a neighbour list with bothways=True.
    So we need check if an edge has already been added.

    TODO:
        Need a better strategy to deal with periodic boundary conditions.
        Currently, we ignore the duplicate edges between the same pair of atoms.

    Args:
        atoms: Atoms object.
        neighlist: NeighborList object.
        indices: Indices of atoms to be included in the graph.

    Returns:
        A graph.

    """
    natoms = len(atoms)
    if indices is None:
        indices = list(range(natoms))

    chemical_symbols = atoms.get_chemical_symbols()

    graph = nx.Graph()
    for i in indices:
        s_i = chemical_symbols[i]
        graph.add_node(f"{s_i}_{i}")

    visitsed_edges = []
    for i in indices:
        nei_indices, nei_offsets = neighlist.get_neighbors(i)
        s_i = chemical_symbols[i]
        for j, o in zip(nei_indices, nei_offsets):
            if j in indices:
                if (i, j) or (j, i) not in visitsed_edges:
                    s_j = chemical_symbols[j]
                    graph.add_edge(
                        f"{s_i}_{i}",
                        f"{s_j}_{j}",
                        bond="{}-{}".format(*sorted([s_i, s_j])),
                        shift=tuple(o.tolist()),
                    )
                    visitsed_edges.append((i, j))
                else:
                    ...  # The edge has already been added.
            else:
                ...  # The neighbour atom should also be in the selected indices.

    return graph


def get_minimum_distance_between_two_fragments(
    atoms: Atoms, frag_a: list[int], frag_b: list[int], neighlist: NeighborList
) -> float:
    """Get the minimum distance between two fragments.

    Args:
        atoms: Atoms object.
        fragment_indices: A list of fragment indices.

    Returns:
        The minimum distance between two fragments.

    """
    pbc = atoms.pbc

    num_atoms_in_a = len(frag_a)

    positions = atoms.get_positions()
    raw_vectors = positions[frag_a] - positions[frag_b][:, np.newaxis, :].repeat(num_atoms_in_a, axis=1)
    raw_vectors = raw_vectors.reshape(-1, 3)
    _, mic_distances = find_mic(v=raw_vectors, cell=atoms.cell, pbc=pbc)

    min_dist = np.min(mic_distances)

    return min_dist


def get_fragments_by_graph(
    atoms: Atoms,
    grp_expr: Optional[str] = None,
    cutoff: Optional[float] = None,
    inter_fragment_distance: Optional[float] = None,
) -> tuple[list[list[int]], list[str]]:
    """Get fragments by graph.

    Args:
        atoms: Atoms object.
        grp_expr: Group expression.
        cutoff: Cutoff distance.
        inter_fragment_distance: Minimum distance between fragments.

    Returns:
        A tuple of fragment indices and fragment formulae.

    """
    # Initialise a neighbour list
    if cutoff is None:
        cutoffs = natural_cutoffs(atoms, mult=1)
    else:
        cutoffs = np.array([cutoff / 2.0] * len(atoms))

    nl = NeighborList(
        cutoffs=cutoffs,
        skin=0.2,
        sorted=False,
        self_interaction=False,
        bothways=True,
    )
    nl.update(atoms)

    # Get cluster graphs and reassemble clusters using DFS
    chemical_symbols = atoms.get_chemical_symbols()

    if grp_expr is not None:
        group_indices = evaluate_group_expression(atoms, grp_expr)
    else:
        group_indices = None

    fragment_indices, fragment_formulae = [], []
    graph = create_a_graph(atoms, neighlist=nl, indices=group_indices)
    for c in nx.connected_components(graph):
        indices = [int(x.split("_")[1]) for x in c]
        fragment_indices.append(indices)
        chem_form = Formula.from_list([chemical_symbols[i] for i in indices]).format("hill")
        fragment_formulae.append(chem_form)
        # Need proper connectivity in fragments?
        # reassemble_fragment_by_depth_first_search(atoms, nl, indices[0])

    # Further check if fragments are loosely connected, if so they will be merged.
    # For example, a H2O molecule stays close with the underlying adsorbate by hydrogen bonds.
    if inter_fragment_distance is not None:
        frag_graph = nx.Graph()
        num_fragments = len(fragment_indices)
        for i in range(num_fragments):
            frag_graph.add_node(i)
        for i in range(num_fragments):
            for j in range(i + 1, num_fragments):
                min_dist = get_minimum_distance_between_two_fragments(
                    atoms, fragment_indices[i], fragment_indices[j], nl
                )
                if min_dist < inter_fragment_distance:
                    frag_graph.add_edge(i, j)
        merged_indices, merged_formulae = [], []
        for c in nx.connected_components(frag_graph):
            merged_indices.append(list(itertools.chain(*[fragment_indices[i] for i in c])))
            merged_formulae.append("+".join([fragment_formulae[i] for i in c]))
        fragment_indices, fragment_formulae = merged_indices, merged_formulae

    return fragment_indices, fragment_formulae


class DissociativeDescriber(BaseDescriber):

    name: str = "dissociative"

    def __init__(
        self,
        max_num_frag: int = 1,
        inter_fragment_distance: Optional[float] = None,
        group: Optional[str] = None,
        *args,
        **kwargs,
    ):
        """Initialise the describer."""
        super().__init__(*args, **kwargs)

        self.max_num_frag = max_num_frag

        self.inter_fragment_distance = inter_fragment_distance  # Ang

        self.group = group

        return

    def run(self, structures):
        """"""
        dissociative_states = []
        for atoms in structures:
            _, fragments = get_fragments_by_graph(
                atoms, grp_expr=self.group, cutoff=None, inter_fragment_distance=self.inter_fragment_distance
            )
            num_fragments = len(fragments)
            if num_fragments > self.max_num_frag:
                dissociative_states.append(True)
            else:
                dissociative_states.append(False)

        return np.array(dissociative_states, dtype=np.int32)


if __name__ == "__main__":
    ...
