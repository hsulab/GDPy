import numbers
from typing import Callable, NamedTuple, Union

import networkx as nx
import numpy as np
from ase import Atom, Atoms

from .data import NeighbourData

ADSORBATE_SUBSTRATE_DISTANCE: float = 2.5  # Angstrom

DIS_SURF2SURF: int = 2
DIS_ADS2SURF: int = 1


class ExpandGraphFunctions(NamedTuple):
    node_id_func: Callable
    add_edge_func: Callable


def expand_grids(grid: Union[int, tuple[int, int, int]]) -> list[tuple[int, int, int]]:
    """Get grids as a list of tuples.

    Args:
        grid: The grid dimension(s) to iterate over (x or (x, y, z))

    Returns:
        tuple: (x, y, z) coordinates
    """
    # Expand to 3D grid
    if isinstance(grid, int):
        grid_ = (grid, grid, grid)
    else:
        grid_ = grid

    grids = []
    for x in range(-grid_[0], grid_[0] + 1):
        for y in range(-grid_[1], grid_[1] + 1):
            for z in range(-grid_[2], grid_[2] + 1):
                grids.append((x, y, z))

    return grids


def node_id_func(atom: Atom, **kwargs):
    """Generate a node ID string for an atom."""
    # No information of ghost atoms is stored in the graph!!
    index = kwargs.get("index")
    assert isinstance(index, numbers.Integral)
    shift = kwargs.get("shift")
    assert isinstance(shift, (tuple, np.ndarray))

    return "{}:{}:[{},{},{}]".format(atom.symbol, index, shift[0], shift[1], shift[2])


def add_edge_func(a_i: Atom, a_j: Atom, **kwargs) -> tuple[str, str, dict]:
    """Generate edge information between two atoms."""
    kw_i, kw_j = kwargs.get("kw_i", {}), kwargs.get("kw_j", {})
    u, v = node_id_func(a_i, **kw_i), node_id_func(a_j, **kw_j)

    group_indices = kwargs.get("group_indices", [])

    i, j = kw_i.get("index"), kw_j.get("index")
    dist = DIS_SURF2SURF - (DIS_ADS2SURF if i in group_indices else 0) - (DIS_ADS2SURF if j in group_indices else 0)

    edge_attrs = dict(
        bond="{}-{}".format(*sorted([a_i.symbol, a_j.symbol])),
        index="{}:{}".format(*sorted([i, j])),
        dist=dist,
        # dist_edge=dis,
        ads_only=0 if (i in group_indices and j in group_indices) else 2,
    )

    return u, v, edge_attrs


def build_expand_graph(
    atoms: Atoms,
    neigh: NeighbourData,
    group_indices: list[int],
    node_id_func=node_id_func,
    add_edge_func=add_edge_func,
    gmax: tuple[int, int, int] = (1, 1, 0),
) -> nx.Graph:
    """Build an expanded graph from ASE Atoms and group indices.

    Args:
        atoms: ASE Atoms object representing the structure.
        group_indices: List of atom indices to include in the graph.
        gmax: Maximum grid expansion in each dimension (x, y, z).

    Returns:
        A NetworkX graph representing the expanded atoms.
    """
    num_atoms = len(atoms)
    all_indices = list(range(num_atoms))

    # Create graph
    graph = nx.Graph()

    # add nodes and edges
    grids = expand_grids(gmax)
    for i in all_indices:
        a_i = atoms[i]
        assert isinstance(a_i, Atom)
        for grid in grids:
            graph.add_node(
                node_id_func(a_i, index=i, shift=grid),
                index=int(i),
                central_ads=False,
            )

    is_edge_in_grid = lambda origin, shift: all(-gmax[d] <= origin[d] + shift[d] <= gmax[d] for d in range(3))
    is_edge_for_ads = lambda d, i, j: d >= ADSORBATE_SUBSTRATE_DISTANCE and (i in group_indices or j in group_indices)

    for i in all_indices:
        a_i = atoms[i]
        mask = neigh.senders == i
        for grid in grids:
            for j, s, d in zip(neigh.receivers[mask], neigh.shifts[mask], neigh.distances[mask]):
                if is_edge_in_grid(grid, s) and not is_edge_for_ads(d, i, j):
                    a_j = atoms[j]
                    u, v, edge_attrs = add_edge_func(
                        a_i,
                        a_j,
                        kw_i={"index": i, "shift": grid},
                        kw_j={"index": j, "shift": tuple(np.array(grid) + s)},
                        group_indices=group_indices,
                    )
                    graph.add_edge(u, v, **edge_attrs)

    return graph


expand_graph_functions = ExpandGraphFunctions(
    node_id_func=node_id_func,
    add_edge_func=add_edge_func,
)
