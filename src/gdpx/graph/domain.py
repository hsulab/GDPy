from typing import Callable, NamedTuple

import networkx as nx
import numpy as np
from ase import Atom, Atoms

from .data import NeighbourData


class NodeID(NamedTuple):
    sym: str
    idx: int
    shift: tuple[int, int, int]


class DomainGraphFunctions(NamedTuple):
    node_id_func: Callable
    add_edge_func: Callable


def canonicalise_shift(shift: np.ndarray | tuple[int, int, int]) -> tuple[int, int, int]:
    """
    Convert any integer-like iterable (possibly np.int32 or np.int64) into a tuple of Python ints.
    This ensures consistent hashing and equality in NetworkX.
    """
    arr = np.asarray(shift, dtype=np.int32)
    assert arr.shape == (3,)
    return int(arr[0]), int(arr[1]), int(arr[2])


def node_id_func(atom: Atom, idx: int, shift: tuple[int, int, int]) -> NodeID:
    """The nodes include both local (shift=(0,0,0)) and ghost atoms."""
    return NodeID(sym=atom.symbol, idx=int(idx), shift=canonicalise_shift(shift))


def add_edge_func(
    a_i: Atom, a_j: Atom, idx_i: int, idx_j: int, shift_i: tuple[int, int, int], shift_j: tuple[int, int, int]
):
    """The ghost atoms are implicitly added when adding edges."""
    u = node_id_func(a_i, idx_i, shift_i)
    v = node_id_func(a_j, idx_j, shift_j)
    s_i, s_j = a_i.symbol, a_j.symbol
    bond = "{}-{}".format(*sorted([s_i, s_j]))
    edge_attrs = {"bond": bond}

    return u, v, edge_attrs


domain_graph_functions = DomainGraphFunctions(
    node_id_func=node_id_func,
    add_edge_func=add_edge_func,
)


def build_domain_graph(
    atoms: Atoms, neigh: NeighbourData, group_indices: list[int], include_neighbors: bool = False
) -> nx.Graph:
    """Build a domain graph from ASE Atoms and neighbor data.

    The nodes include both local (shift=(0,0,0)) and ghost atoms.
    The ghost atoms are implicitly added when adding edges.

    Args:
        atoms: ASE Atoms object representing the structure.
        neigh: Neighbor data containing senders, receivers, distances, and shifts.
        group_indices: List of atom indices to include in the graph.

    Returns:
        A NetworkX graph representing the domain.

    """
    # Create graph
    graph = nx.Graph()

    # add nodes and edges
    chemical_symbols = atoms.get_chemical_symbols()
    for i in group_indices:
        graph.add_node(
            NodeID(chemical_symbols[i], int(i), canonicalise_shift((0, 0, 0))),
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
            s_i, s_j = chemical_symbols[i], chemical_symbols[j]
            bond = "{}-{}".format(*sorted([s_i, s_j]))
            u = NodeID(sym=s_i, idx=int(i), shift=canonicalise_shift((0, 0, 0)))
            v = NodeID(sym=s_j, idx=int(j), shift=canonicalise_shift(s))
            graph.add_edge(u, v, bond=bond)
            used_pairs.add(pair)

    return graph
