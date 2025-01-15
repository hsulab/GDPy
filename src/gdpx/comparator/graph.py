#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from typing import Optional

import networkx as nx
import numpy as np
from ase import Atoms
from ase.neighborlist import NeighborList, natural_cutoffs
from joblib import Parallel, delayed

from gdpx.group.group import create_a_group

from ..utils.command import CustomTimer
from .comparator import AbstractComparator

bond_match = nx.algorithms.isomorphism.categorical_edge_match("bond", "")


def create_a_graph(atoms, indices=None):
    """"""
    natoms = len(atoms)
    if indices is None:
        indices = range(natoms)

    graph = nx.Graph()
    for i in indices:
        graph.add_node(
            atoms.symbols[i] + "_" + str(i),
        )

    nl = NeighborList(
        cutoffs=natural_cutoffs(atoms, mult=1),
        skin=0.2,
        sorted=False,
        self_interaction=False,
        bothways=False,
    )
    nl.update(atoms)
    for i in indices:  # TODO: PBC
        nei_indices, nei_offsets = nl.get_neighbors(i)
        s_i = atoms.symbols[i]
        for j in nei_indices:
            s_j = atoms.symbols[j]
            graph.add_edge(
                f"{s_i}_{i}", f"{s_j}_{j}", bond="{}{}".format(*sorted([s_i, s_j]))
            )

    return graph


# Function to calculate the inertia tensor for a point mass
def point_mass_inertia_tensor(mass, position):
    I = np.zeros((3, 3))
    r_squared = np.dot(position, position)

    for i in range(3):
        for j in range(3):
            if i == j:
                I[i, j] = mass * (r_squared - position[i] ** 2)
            else:
                I[i, j] = -mass * position[i] * position[j]

    return I


# Function to calculate the total inertia tensor for the nanoparticle
def calculate_inertia_tensor(coordinates, atomic_masses):
    total_inertia_tensor = np.zeros((3, 3))

    # Iterate through each copper atom and add its contribution to the total inertia tensor
    for i in range(len(coordinates)):
        atom_position, atomic_mass = coordinates[i], atomic_masses[i]
        atom_inertia_tensor = point_mass_inertia_tensor(atomic_mass, atom_position)
        total_inertia_tensor += atom_inertia_tensor

    return total_inertia_tensor


class GraphComparator(AbstractComparator):

    group: Optional[str] = None

    def __init__(self, mic=True, group=None, *args, **kwargs):
        """"""
        super().__init__(*args, **kwargs)

        self.mic = mic
        self.group = group

        return

    @staticmethod
    def _process_single_structure(atoms, group):
        """"""
        group_indices = create_a_group(atoms, group)
        graph = create_a_graph(atoms, group_indices)

        return graph

    def prepare_data(self, frames: list[Atoms]):
        """"""
        with CustomTimer(name="graph", func=self._debug):
            graphs = Parallel(n_jobs=self.njobs)(
                delayed(self._process_single_structure)(atoms, self.group)
                for atoms in frames
            )

        return graphs

    def looks_like(self, fp1, fp2):
        """"""
        is_isomorphic = nx.algorithms.isomorphism.is_isomorphic(
            fp1, fp2, edge_match=bond_match
        )

        return is_isomorphic

    def _looks_like_with_atoms(self, a1, a2):
        """"""
        is_similar = self.compare_composition(a1, a2)
        if is_similar:
            # -
            ainds = None
            if self.group is not None:
                g1 = create_a_group(a1, self.group)
                g2 = create_a_group(a2, self.group)
                if g1 == g2:  # can be []
                    ainds = g1
                else:
                    ainds = []
                if len(ainds) > 0:
                    self._print(f"natoms: {len(ainds)}")
                    self._print(f"{a1[ainds].get_chemical_formula()}")
                    # write("xxx.xyz", a1[ainds])
                else:
                    ainds = range(len(a1))  # atomic indices
            else:
                ainds = range(len(a1))  # atomic indices
            # -
            graph_1 = create_a_graph(a1, ainds)
            graph_2 = create_a_graph(a2, ainds)
            # matcher = nx.algorithms.isomorphism.GraphMatcher(
            #     graph_1, graph_2, edge_match=bond_match
            # )
            # is_isomorphic = matcher.is_isomorphic()
            is_isomorphic = nx.algorithms.isomorphism.is_isomorphic(
                graph_1, graph_2, edge_match=bond_match
            )
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


if __name__ == "__main__":
    ...
