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


def extract_chemical_environments(
    graph: nx.Graph, atoms: Atoms, group_indices: list[int], graph_radius: int
) -> list[nx.Graph]:
    """Extract chemical environments from the graph.

    Args:
        graph: The input graph.
        atoms: ASE Atoms object representing the structure.
        group_indices: List of atom indices to extract environments for.
        graph_radius: The radius of the chemical environment.

    Returns:
        A list of subgraphs representing the chemical environments.
    """

    # Get nodes corresponding to group_indices (single atom or molecule)
    group_nodes = [node_id_func(atoms[i], index=i, shift=(0, 0, 0)) for i in group_indices]  # type: ignore
    group_graph = nx.subgraph(graph, group_nodes)

    # nx.connected_component_subgraphs removed in v2.4
    cluster_graphs = [group_graph.subgraph(c) for c in nx.connected_components(group_graph)]

    chemical_environments = []
    for _, cluster in enumerate(cluster_graphs):
        start_node = list(cluster.nodes)[0]
        cluster = nx.ego_graph(graph, start_node, radius=0, distance="ads_only")
        chem_env = nx.ego_graph(
            graph, start_node, radius=(graph_radius * DIS_SURF2SURF) + DIS_ADS2SURF, distance="dist"
        )

        # update attrs
        for node in cluster.nodes():
            chem_env.add_node(node, central_ads=True)  # node within (0,0,0) grid

        for node in cluster.nodes():
            chem_env.add_node(node, ads=True)  # node within cluster

        chemical_environments.append(chem_env)

    return chemical_environments


#: Handles isomorphism for bonds.
edge_match = nx.algorithms.isomorphism.categorical_edge_match("bond", "")

#: Handles isomorphism for atoms with regards to perodic boundary conditions.
node_match = nx.algorithms.isomorphism.categorical_node_match(["index", "ads"], [-1, False])


def get_unique_chemical_environments_by_bonds(chem_envs: list[nx.Graph]) -> list[int]:
    """Find unique environments based on graph edges.

    This method compares one graph with another.

    Args:
        chem_envs: A list of graph representation of an atom.

    Returns:
        Indices of unique graphs.

    """

    nsites = len(chem_envs)

    unique_indices = [0]
    for i in range(1, nsites):
        for j in unique_indices:
            env_i, env_j = chem_envs[i], chem_envs[j]
            if nx.algorithms.isomorphism.is_isomorphic(env_i, env_j, edge_match=edge_match):
                break
        else:
            unique_indices.append(i)

    return unique_indices


def compare_chemical_environments_within_two_groups(chem_envs1: list[nx.Graph], chem_envs2: list[nx.Graph]) -> bool:
    """Compares two sets of chemical environments to see if they are one-to-one matching.

    Useful for detecting if two sets of adsorbates are in the same configurations.

    Args:
        chem_envs1: A list of chemical environments including duplicate adsorbates
        chem_envs2: A list of chemical environments, including duplicate adsorbates

    Returns:
        bool: Is there a matching graph (site / adsorbate) for each graph.

    """
    if len(chem_envs1) != len(chem_envs2):
        return False

    used = [False] * len(chem_envs2)

    for e1 in chem_envs1:
        found = False
        for j, e2 in enumerate(chem_envs2):
            if used[j]:
                continue
            if nx.algorithms.isomorphism.is_isomorphic(e1, e2, edge_match=edge_match):
                used[j] = True
                found = True
                break
        if not found:
            return False

    assert all(used)

    return True


def get_unique_chemical_environments_among_goups(chem_envs_groups: list[list[nx.Graph]]):
    """Given a list of chemical environments, find the unique environments and keep track of metadata if required.

    This function exists largely to help with unique site detection
    but its performance will scale badly with extremely large numbers
    of chemical environments to check.  This may be split into parallel
    jobs.

    Args:
        chem_env_groups: Chemical environments to compare against each other

    Returns:
        list[list[list[networkx.Graph]]]: A list of unique chemical environments
                                          with their duplicates
        list[list[object]]: A matching list of metadata
    """
    if len(chem_envs_groups) == 0:
        return [[], []]

    # Keep track of known unique environments
    unique_env_groups = []

    for i, env_group in enumerate(chem_envs_groups):
        for _, (unique_indices, unique_env_group) in enumerate(unique_env_groups):
            if compare_chemical_environments_within_two_groups(env_group, unique_env_group):
                unique_indices.append(i)
                break
        else:  # was unique
            unique_env_groups.append(([i], env_group))

    # Zip trick to split into two lists to return
    return zip(*[(env, indices) for (indices, env) in unique_env_groups])


expand_graph_functions = ExpandGraphFunctions(
    node_id_func=node_id_func,
    add_edge_func=add_edge_func,
)
