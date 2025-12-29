from typing import Optional

import networkx as nx
import numpy as np
from ase import Atoms
from joblib import Parallel, delayed

from gdpx.graph.base import AtomicGraph
from gdpx.group import evaluate_group_expression
from gdpx.utils.profiler import CustomTimer

from .comparator import BaseComparator

bond_match = nx.algorithms.isomorphism.categorical_edge_match("bond", "")


def build_graph(
    atoms: Atoms, indices: Optional[list[int]] = None, ignored_bonds: Optional[list[str]] = None
) -> nx.Graph:
    """"""
    indices = indices if indices is not None else list(range(len(atoms)))

    graph_builder = AtomicGraph(atoms, graph_type="partial")
    graph_builder.build(indices=indices, ratio=1.0, skin=0.2, include_neighbors=True)

    # Remove edges by ignored bonds
    graph = graph_builder.graph
    assert isinstance(graph, nx.Graph)
    if ignored_bonds is not None:
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
        nodes_to_remove = [
            u for u in graph.nodes() if u not in node_ids_in_edges and int(u.split("_")[-1]) not in indices
        ]
        graph.remove_nodes_from(nodes_to_remove)

    return graph


def point_mass_inertia_tensor(mass, position):
    """Function to calculate the inertia tensor for a point mass."""
    I = np.zeros((3, 3))
    r_squared = np.dot(position, position)

    for i in range(3):
        for j in range(3):
            if i == j:
                I[i, j] = mass * (r_squared - position[i] ** 2)
            else:
                I[i, j] = -mass * position[i] * position[j]

    return I


def calculate_inertia_tensor(coordinates, atomic_masses):
    """Function to calculate the total inertia tensor for the nanoparticle."""
    total_inertia_tensor = np.zeros((3, 3))

    # Iterate through each copper atom and add its contribution to the total inertia tensor
    for i in range(len(coordinates)):
        atom_position, atomic_mass = coordinates[i], atomic_masses[i]
        atom_inertia_tensor = point_mass_inertia_tensor(atomic_mass, atom_position)
        total_inertia_tensor += atom_inertia_tensor

    return total_inertia_tensor


class GraphComparator(BaseComparator):
    def __init__(
        self, group: Optional[str] = None, ignored_pairs: Optional[list[tuple[str, str]]] = None, *args, **kwargs
    ):
        """Initialise the comparator.

        Args:
            group: The group expression to select atoms to build graph for comparison.

        """
        super().__init__(*args, **kwargs)

        self.group = group

        ignore_pairs_ = []
        if ignored_pairs is not None:
            for s1, s2 in ignored_pairs:
                ignore_pairs_.append("-".join([s1, s2]))
                ignore_pairs_.append("-".join([s2, s1]))
        self.ignored_pairs = ignore_pairs_

        return

    @staticmethod
    def _process_single_structure(atoms: Atoms, group: str, ignored_pairs: list[str]) -> nx.Graph:
        """"""
        group_indices = evaluate_group_expression(atoms, group)
        graph = build_graph(atoms, group_indices, ignored_pairs)

        return graph

    def prepare_data(self, frames: list[Atoms]):
        """"""
        with CustomTimer(name="creating graphs", func=self._print):
            graphs = Parallel(n_jobs=self.njobs)(
                delayed(self._process_single_structure)(atoms, self.group, self.ignored_pairs) for atoms in frames
            )

        return graphs

    def looks_like(self, a1: Atoms, a2: Atoms) -> bool:
        """"""
        fingerprints = self.prepare_data([a1, a2])
        fp1, fp2 = fingerprints
        is_similar = self.compare_fingerprints(fp1, fp2)

        return is_similar

    def compare_fingerprints(self, fp1, fp2):
        """"""
        is_isomorphic = nx.algorithms.isomorphism.is_isomorphic(fp1, fp2, edge_match=bond_match)

        return is_isomorphic

    def _looks_like_with_atoms(self, a1, a2):
        """"""
        is_similar = self.compare_composition(a1, a2)
        if is_similar:
            group_indices = list(range(len(a1)))  # number of atoms have been checked to be the same
            if self.group is not None:
                g1 = evaluate_group_expression(a1, self.group)
                g2 = evaluate_group_expression(a2, self.group)
                if g1 == g2:  # can be []
                    group_indices = g1
                else:
                    group_indices = []
                if len(group_indices) > 0:
                    self._print(f"natoms: {len(group_indices)}")
                    self._print(f"{a1[group_indices].get_chemical_formula()}")
                    # write("xxx.xyz", a1[ainds])
                else:
                    ...
            else:
                ...
            # Create graphs
            graph_1 = build_graph(a1, group_indices)
            graph_2 = build_graph(a2, group_indices)
            # matcher = nx.algorithms.isomorphism.GraphMatcher(
            #     graph_1, graph_2, edge_match=bond_match
            # )
            # is_isomorphic = matcher.is_isomorphic()
            is_isomorphic = nx.algorithms.isomorphism.is_isomorphic(graph_1, graph_2, edge_match=bond_match)
            self._print(f"  isomorphic: {is_isomorphic}")
            if is_isomorphic:
                ...
                # inertia_1 = calculate_inertia_tensor(
                #    a1.positions[ainds, :], a1.get_masses()[ainds]
                # )
                # eig_1 = np.linalg.eigvals(inertia_1)
                # inertia_2 = calculate_inertia_tensor(
                #    a2.positions[ainds, :], a2.get_masses()[ainds]
                # )
                # eig_2 = np.linalg.eigvals(inertia_2)
                # self._print("  "+str(eig_1 - eig_2))
                # self._print("  "+str(np.linalg.norm(eig_1 - eig_2)))
                # if np.linalg.norm(eig_1 - eig_2) >= 40.:
                #    is_similar = False
            else:
                is_similar = False
        else:
            ...
        self._print(f" similar: {is_similar}")

        return is_similar
