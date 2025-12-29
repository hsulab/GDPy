from typing import NamedTuple

import networkx as nx
import numpy as np
from ase import Atoms

from .base import NeighbourData


class NodeID(NamedTuple):
    idx: int
    shift: tuple[int, int, int]


def canonicalize_shift(shift: np.ndarray | tuple[int, int, int]) -> tuple[int, int, int]:
    """
    Convert any integer-like iterable (possibly np.int32 or np.int64) into a tuple of Python ints.
    This ensures consistent hashing and equality in NetworkX.
    """
    arr = np.asarray(shift, dtype=np.int32)
    assert arr.shape == (3,)
    return int(arr[0]), int(arr[1]), int(arr[2])


def build_domain_graph(
    atoms: Atoms, neigh: NeighbourData, group_indices: list[int], grids: list[tuple[int, int, int]]
) -> nx.Graph:
    """"""
    group_indices = sorted(group_indices)

    box = atoms.get_cell(complete=True)

    senders, receivers, distances, shifts = neigh.senders, neigh.receivers, neigh.distances, neigh.shifts

    graph = nx.Graph()

    for grid in grids:
        for i in group_indices:
            graph.add_node(
                NodeID(int(i), canonicalize_shift(grid)),
            )

    used_pairs = set()
    for i, j, d, s in zip(senders, receivers, distances, shifts):
        pair = tuple(sorted([i, j]))
        if (i in group_indices and j in group_indices) and (i != j) and pair not in used_pairs:
            u = NodeID(idx=int(i), shift=canonicalize_shift((0, 0, 0)))
            v = NodeID(idx=int(j), shift=canonicalize_shift(s))
            cart_shift = s @ box
            graph.add_edge(u, v, distance=d, shift=cart_shift)
            used_pairs.add(pair)

    return graph
