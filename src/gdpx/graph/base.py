import copy
import itertools
from typing import NamedTuple, Optional

import ase.data
import networkx as nx
import numpy as np
from ase import Atoms
from ase.neighborlist import neighbor_list

from .domain import build_domain_graph


class NeighbourData(NamedTuple):
    senders: np.ndarray
    receivers: np.ndarray
    distances: np.ndarray
    shifts: np.ndarray


def get_bond_distance_dict(atoms: Atoms, ratio: float = 1.02, skin: float = 0.0) -> dict[tuple[int, int], float]:
    """"""
    chemical_symbols = atoms.get_chemical_symbols()
    bond_pairs = itertools.combinations_with_replacement(set(chemical_symbols), 2)
    bond_distance_dict = {}
    for s1, s2 in bond_pairs:
        n1, n2 = ase.data.atomic_numbers[s1], ase.data.atomic_numbers[s2]
        r1, r2 = ase.data.covalent_radii[n1], ase.data.covalent_radii[n2]
        bond_distance_dict[(n1, n2)] = (r1 + r2) * ratio + skin * 2
        bond_distance_dict[(n2, n1)] = (r1 + r2) * ratio + skin * 2

    return bond_distance_dict


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


def rebuild_cluster_by_depth_first_search(
    atoms: Atoms, group_indices: list[int], start_indices: list[int], neigh: NeighbourData
) -> None:
    """Ensure that atoms in the given cluster groups have proper connectivities.

    This uses a neighbour list with bothways=True.
    The start_indices must be from separate connected components (clusters).

    Args:
        atoms: The ASE Atoms object to be modified in place.
        group_indices: The indices of atoms in the cluster.
        start_indices: The starting indices for depth-first search.
        neigh: The neighbour data containing senders, receivers, distances, and shifts.

    Returns:
        None

    """
    senders, receivers, shifts = neigh.senders, neigh.receivers, neigh.shifts
    neigh_idx_dict: dict[int, list[int]] = {i: [] for i in group_indices}
    neigh_sft_dict: dict[int, list[tuple[float, float, float]]] = {i: [] for i in group_indices}
    for i, j, d in zip(senders, receivers, shifts):
        if i in group_indices and j in group_indices:
            neigh_idx_dict[i].append(j)
            neigh_sft_dict[i].append(d)

    def dfs_stack(c: np.ndarray, prev_positions: np.ndarray, positions: np.ndarray, start_indices: list[int]):
        """"""
        visited = set()

        stack = [(i, positions[i]) for i in start_indices]
        while stack:
            i, p = stack.pop()
            if i in visited:
                continue
            visited.add(i)

            nei_indices, nei_shifts = neigh_idx_dict[i], neigh_sft_dict[i]
            for j, s in zip(nei_indices, nei_shifts):
                if j not in visited:
                    # Calculate the distance vector considering periodic boundary conditions
                    v = prev_positions[j] + np.dot(s, c) - p
                    v -= np.dot(np.round(v / c.diagonal()), c)
                    positions[j] = p + v
                    stack.append((j, positions[j]))

        return positions

    positions = dfs_stack(atoms.cell.array, atoms.positions, copy.deepcopy(atoms.positions), start_indices)

    atoms.positions = positions

    return


def prune_neighbour_data_by_bond_distance(
    atoms: Atoms,
    neigh: NeighbourData,
    bond_distance_dict: dict[tuple[int, int], float],
) -> NeighbourData:
    """"""
    chemical_symbols = atoms.get_chemical_symbols()
    senders, receivers, distances, shifts = [], [], [], []
    for i, j, d, s in zip(neigh.senders, neigh.receivers, neigh.distances, neigh.shifts):
        s_i, s_j = chemical_symbols[i], chemical_symbols[j]
        n_i, n_j = ase.data.atomic_numbers[s_i], ase.data.atomic_numbers[s_j]
        if d <= bond_distance_dict[(n_i, n_j)]:
            senders.append(i)
            receivers.append(j)
            distances.append(d)
            shifts.append(s)
    senders = np.array(senders, dtype=int)
    receivers = np.array(receivers, dtype=int)
    distances = np.array(distances, dtype=float)
    shifts = np.array(shifts, dtype=int)

    return NeighbourData(senders, receivers, distances, shifts)


def prune_graph_by_ignored_bonds(
    graph: nx.Graph,
    indices: list[int],
    ignored_bonds: list[str],
) -> nx.Graph:
    """"""
    edges_to_remove = []
    for u, v, data in graph.edges(data=True):
        bond = data.get("bond", "")
        if bond in ignored_bonds:
            edges_to_remove.append((u, v))
    graph.remove_edges_from(edges_to_remove)
    # we keep only nodes that are in edges or in indices
    node_ids_in_edges = set()
    for u, v in graph.edges():
        node_ids_in_edges.add(u)
        node_ids_in_edges.add(v)
    nodes_to_remove = [u for u in graph.nodes() if u not in node_ids_in_edges and int(u.split("_")[-1]) not in indices]
    graph.remove_nodes_from(nodes_to_remove)

    return graph


class AtomicGraph:
    def __init__(self, atoms: Atoms, graph_type: str = "partial"):
        """"""
        match graph_type:
            case "partial":
                self._build = build_partial_graph
                self.self_interaction = False
            case "domain":
                self._build = build_domain_graph
                self.self_interaction = True
            case _:
                raise Exception(f"Unknown graph building method `{graph_type}`.")

        self._atoms: Atoms = atoms
        self._graph: Optional[nx.Graph] = None

        return

    def build(
        self,
        indices: Optional[list[int]] = None,
        cutoff: Optional[float] = None,
        ratio: float = 1.03,
        skin: float = 0.0,
        include_neighbors: bool = False,
        ignored_bonds: Optional[list[str]] = None,
    ) -> None:
        """"""
        indices = indices if indices is not None else list(range(len(self._atoms)))

        if cutoff is None:
            bond_distance_dict = get_bond_distance_dict(self._atoms, ratio=ratio, skin=skin)
            cutoff = max(bond_distance_dict.values())
            self._neigh = prune_neighbour_data_by_bond_distance(
                self._atoms,
                NeighbourData(
                    *neighbor_list("ijdS", self._atoms, cutoff=cutoff, self_interaction=self.self_interaction)
                ),
                bond_distance_dict=bond_distance_dict,
            )
        else:
            self._neigh = NeighbourData(
                *neighbor_list("ijdS", self._atoms, cutoff=cutoff, self_interaction=self.self_interaction)
            )

        self._graph = self._build(
            self._atoms,
            self._neigh,
            group_indices=indices,
            include_neighbors=include_neighbors,
        )
        if ignored_bonds is not None:
            self._graph = prune_graph_by_ignored_bonds(
                self._graph,
                indices=indices,
                ignored_bonds=ignored_bonds,
            )

        return

    @property
    def graph(self) -> Optional[nx.Graph]:
        """"""
        return self._graph

    def get_cluster_indices(self) -> list[list[int]]:
        """"""
        assert self._graph is not None, "Graph has not been built yet."
        cluster_groups = []
        for component in nx.connected_components(self._graph):
            cluster_indices = [int(node.split("_")[-1]) for node in component]
            cluster_groups.append(cluster_indices)

        return cluster_groups

    def get_clusters(self, rebuild: bool = False) -> list[Atoms]:
        """"""
        assert self._graph is not None, "Graph has not been built yet."
        cluster_groups = self.get_cluster_indices()

        if rebuild:
            atoms = copy.deepcopy(self._atoms)
            rebuild_cluster_by_depth_first_search(
                atoms,
                group_indices=list(itertools.chain.from_iterable(cluster_groups)),
                start_indices=[indices[0] for indices in cluster_groups],
                neigh=self._neigh,
            )
        else:
            atoms = self._atoms

        clusters = []
        for cluster_indices in cluster_groups:
            cluster = atoms[cluster_indices]  # getitem makes deep copy
            assert isinstance(cluster, Atoms)
            cluster.info["_host_indices"] = cluster_indices
            clusters.append(cluster)

        return clusters
