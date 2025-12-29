import numbers
from typing import Callable, NamedTuple

import networkx as nx
from ase import Atom, Atoms

from .data import NeighbourData


class PartialGraphFunctions(NamedTuple):
    node_id_func: Callable
    add_edge_func: Callable


def node_id_func(atom: Atom, **kwargs) -> str:
    """Generate a node ID string for an atom."""
    # No information of ghost atoms is stored in the graph!!
    index = kwargs.get("index")
    assert isinstance(index, numbers.Integral)

    return f"{atom.symbol}_{int(index)}"


def add_edge_func(a_i: Atom, a_j: Atom, **kwargs) -> tuple[str, str, dict]:
    """Generate edge information between two atoms."""
    # If the box is too small, the edge between two atoms may appear multiple times with different shifts,
    # which are not considered here.
    kw_i = kwargs.get("kw_i", {})
    kw_j = kwargs.get("kw_j", {})
    u = node_id_func(a_i, **kw_i)
    v = node_id_func(a_j, **kw_j)

    s_i, s_j = a_i.symbol, a_j.symbol
    bond = "{}-{}".format(*sorted([s_i, s_j]))
    shift_j = kw_j.get("shift")
    edge_attrs = {"bond": bond, "shift": shift_j}

    return u, v, edge_attrs


partial_graph_functions = PartialGraphFunctions(
    node_id_func=node_id_func,
    add_edge_func=add_edge_func,
)


def build_partial_graph(
    atoms: Atoms,
    neigh: NeighbourData,
    group_indices: list[int],
    include_neighbors: bool = False,
) -> nx.Graph:
    """Build graph from partial atoms.

    Nodes are created only for the given indices.
    Edges are created only between the given indices if include_neighbors is False.

    Args:
        atoms: The ASE Atoms object.
        neigh: The neighbour data containing senders, receivers, distances, and shifts.
        group_indices: The indices of atoms to include in the graph. If None, include all atoms.
        include_neighbors: Whether to include edges to neighboring atoms outside.

    """
    # Create graph
    graph = nx.Graph()

    # add nodes and edges
    chemical_symbols = atoms.get_chemical_symbols()
    for i in group_indices:
        graph.add_node(chemical_symbols[i] + "_" + str(i))

    is_edge_valid = (
        lambda i, j: (i in group_indices and j in group_indices)
        if not include_neighbors
        else (i in group_indices or j in group_indices)
    )

    used_pairs = set()
    for i, j, s in zip(neigh.senders, neigh.receivers, neigh.shifts):
        pair = tuple(sorted([i, j]))
        if is_edge_valid(i, j) and (i != j) and pair not in used_pairs:
            # No information of ghost atoms is stored in the graph!!
            # If the box is too small, the edge between two atoms may appear multiple times with different shifts,
            # which are not considered here.
            s_i, s_j = chemical_symbols[i], chemical_symbols[j]
            bond = "{}-{}".format(*sorted([s_i, s_j]))
            graph.add_edge(f"{s_i}_{i}", f"{s_j}_{j}", bond=bond, shift=s)
            used_pairs.add(pair)

    return graph
