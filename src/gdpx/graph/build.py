from typing import Callable

import networkx as nx
from ase import Atom, Atoms

from .data import NeighbourData


def build_atomic_graph(
    atoms: Atoms,
    neigh: NeighbourData,
    group_indices: list[int],
    node_id_func: Callable,
    add_edge_func: Callable,
    include_neighbors: bool = False,
) -> nx.Graph:
    """Build a graph from ASE Atoms and neighbor data.

    Args:
        atoms: ASE Atoms object representing the structure.
        neigh: Neighbor data containing senders, receivers, distances, and shifts.
        group_indices: List of atom indices to include in the graph.

    Returns:
        A NetworkX graph representing the atoms.

    """
    # Create graph
    graph = nx.Graph()

    # add nodes and edges
    for i in group_indices:
        a_i = atoms[i]
        assert isinstance(a_i, Atom)
        graph.add_node(
            # NodeID(chemical_symbols[i], int(i), canonicalise_shift((0, 0, 0))),
            node_id_func(a_i, idx=i, shift=(0, 0, 0)),
        )

    is_edge_valid = (
        lambda i, j: (i in group_indices and j in group_indices)
        if not include_neighbors
        else (i in group_indices or j in group_indices)
    )

    used_pairs = set()
    for i, j, s in zip(neigh.senders, neigh.receivers, neigh.shifts):
        pair = tuple(sorted([i, j]))
        if is_edge_valid(i, j) and (i != j) and pair not in used_pairs:
            a_i, a_j = atoms[i], atoms[j]
            assert isinstance(a_i, Atom)
            assert isinstance(a_j, Atom)
            # s_i, s_j = a_i.symbol, a_j.symbol
            # bond = "{}-{}".format(*sorted([s_i, s_j]))
            # u = NodeID(sym=s_i, idx=int(i), shift=canonicalise_shift((0, 0, 0)))
            # v = NodeID(sym=s_j, idx=int(j), shift=canonicalise_shift(s))
            # graph.add_edge(u, v, bond=bond)
            u, v, edge_attrs = add_edge_func(a_i, a_j, idx_i=i, idx_j=j, shift_i=(0, 0, 0), shift_j=s)
            graph.add_edge(u, v, **edge_attrs)
            used_pairs.add(pair)

    return graph
