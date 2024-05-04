#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import copy
import dataclasses
import itertools
from typing import List, Mapping

import networkx as nx
import numpy as np
from ase import Atoms
from ase.data import atomic_masses, atomic_numbers
from ase.formula import Formula
from ase.neighborlist import NeighborList, natural_cutoffs

from .utils import grid_iterator, node_symbol, unpack_node_name


def find_product(
    atoms: Atoms, reactants: List[List[int]], grid=[1, 1, 0], radii_multi=1.0, skin=0.0
) -> List[List[int]]:
    """Find if there were a product from input reactants."""
    valid_indices = list(itertools.chain.from_iterable(reactants))

    # - create local graph
    covalent_radii = natural_cutoffs(atoms, radii_multi)
    nl = NeighborList(
        covalent_radii, skin=skin, sorted=False, self_interaction=False, bothways=True
    )
    nl.update(atoms)

    # print([covalent_radii[i] for i in valid_indices])

    graph = nx.Graph()

    # grid = [1,1,0] # for surface
    # -- add nodes
    for centre_idx in valid_indices:
        for x, y, z in grid_iterator(grid):
            graph.add_node(
                node_symbol(atoms[centre_idx].symbol, centre_idx, (x, y, z)),
                index=centre_idx,
            )

    # -- add edges
    for centre_idx in valid_indices:
        for x, y, z in grid_iterator(grid):
            nei_indices, nei_offsets = nl.get_neighbors(centre_idx)
            for nei_idx, offset in zip(nei_indices, nei_offsets):
                if nei_idx in valid_indices:
                    # NOTE: check if neighbour is in the grid space
                    #       this is not the case when cutoff is too large
                    ox, oy, oz = offset
                    if not (-grid[0] <= ox + x <= grid[0]):
                        continue
                    if not (-grid[1] <= oy + y <= grid[1]):
                        continue
                    if not (-grid[2] <= oz + z <= grid[2]):
                        continue
                    # ---
                    graph.add_edge(
                        node_symbol(atoms[centre_idx].symbol, centre_idx, (x, y, z)),
                        node_symbol(
                            atoms[nei_idx].symbol, nei_idx, (x + ox, y + oy, z + oz)
                        ),
                    )
                else:
                    ...

    # plot_graph(graph, "xxx.png")

    # - find products
    reax_nodes = [node_symbol(atoms[i].symbol, i, (0, 0, 0)) for i in valid_indices]
    reax_graphs = nx.subgraph(graph, reax_nodes)

    prod_graphs = [
        reax_graphs.subgraph(c) for c in nx.connected_components(reax_graphs)
    ]

    products = [[unpack_node_name(u)[1] for u in g.nodes()] for g in prod_graphs]

    return products


@dataclasses.dataclass
class GraphNodeName:

    #: Atom symbol.
    symbol: str

    #: Atom index.
    index: int


class MolecularAdsorbate:

    def __init__(
        self,
        chemical_symbols,
        atomic_indices,
        positions,
        atomic_shifts,
        bonds,
    ):
        """"""
        self._chemical_formula = Formula.from_list(chemical_symbols).format("hill")
        self._chemical_symbols = chemical_symbols
        self._atomic_indices = atomic_indices
        self._atomic_shifts = atomic_shifts
        self._positions = positions
        self._bonds = bonds

        return

    @staticmethod
    def from_graph(atoms, graph) -> "MolecularAdsorbate":
        """"""
        atomic_symbols, atomic_indices, atomic_shifts = [], [], []
        for u in graph.nodes():
            sym, idx, sft = unpack_node_name(u)
            atomic_symbols.append(sym)
            atomic_indices.append(idx)
            atomic_shifts.append(sft)

        bonds = []
        for v in graph.edges():
            bond = (unpack_node_name(v[0])[1], unpack_node_name(v[1])[1])
            bonds.append(bond)

        atomic_shifts = np.array(atomic_shifts) @ atoms.cell

        positions = copy.deepcopy(atoms.positions[atomic_indices])

        return MolecularAdsorbate(
            atomic_symbols,
            atomic_indices,
            positions,
            atomic_shifts,
            bonds,
        )

    @staticmethod
    def from_dict(inp) -> "MolecularAdsorbate":
        """"""

        return MolecularAdsorbate(
            chemical_symbols=inp["chemical_symbols"],
            atomic_indices=inp["atomic_indices"],
            positions=np.array(inp["positions"]),
            atomic_shifts=np.array(inp["atomic_shifts"]),
            bonds=inp["bonds"],
        )

    def get_center_of_mass(self):
        """"""
        masses = np.array(
            [atomic_masses[atomic_numbers[s]] for s in self.chemical_symbols]
        )
        shifted_positions = self.positions + self.atomic_shifts

        com = masses @ shifted_positions / masses.sum()

        return com

    @property
    def bonds(self):
        """"""
        return self._bonds

    @property
    def atomic_indices(self):
        """"""

        return self._atomic_indices

    @property
    def atomic_shifts(self):
        """"""

        return self._atomic_shifts

    # @property
    # def cell(self):
    #     """"""
    #
    #     return self._cell

    @property
    def positions(self):
        """"""

        return self._positions

    @property
    def chemical_formula(self):
        """"""
        # FIXME: Use smiles?

        return self._chemical_formula

    @property
    def chemical_symbols(self):
        """"""

        return self._chemical_symbols

    def as_dict(self):
        """"""
        data = dict(
            atomic_indices=self.atomic_indices,
            chemical_symbols=self.chemical_symbols,
            positions=self.positions.tolist(),
            atomic_shifts=self.atomic_shifts.tolist(),
            bonds=self.bonds,
        )

        return data

    def __repr__(self):
        """"""

        return f"{self.chemical_formula}_{self.atomic_indices}"


def find_molecules(
    atoms: Atoms,
    reactive_indices: List[int],
    grid=[1, 1, 0],
    radii_multi: float = 1.0,
    skin=0.0,
) -> Mapping[str, List[List[int]]]:
    """Find molecules by graph."""
    # valid_indices = list(chain.from_iterable(reactants))

    # add a neighlist
    covalent_radii = natural_cutoffs(atoms, radii_multi)
    nl = NeighborList(
        covalent_radii, skin=skin, sorted=False, self_interaction=False, bothways=True
    )
    nl.update(atoms)

    # create a graph
    graph = nx.Graph()

    for centre_idx in reactive_indices:
        for x, y, z in grid_iterator(grid):
            if x == 0 and y == 0 and z == 0:
                graph.add_node(
                    node_symbol(atoms[centre_idx].symbol, centre_idx, (x, y, z)),
                    index=centre_idx,
                    is_real=True,
                )
            else:
                graph.add_node(
                    node_symbol(atoms[centre_idx].symbol, centre_idx, (x, y, z)),
                    index=centre_idx,
                    is_real=False,
                )

    for centre_idx in reactive_indices:
        for x, y, z in grid_iterator(grid):
            nei_indices, nei_offsets = nl.get_neighbors(centre_idx)
            for nei_idx, offset in zip(nei_indices, nei_offsets):
                if nei_idx in reactive_indices:
                    # NOTE: check if neighbour is in the grid space
                    #       this is not the case when cutoff is too large
                    ox, oy, oz = offset
                    if not (-grid[0] <= ox + x <= grid[0]):
                        continue
                    if not (-grid[1] <= oy + y <= grid[1]):
                        continue
                    if not (-grid[2] <= oz + z <= grid[2]):
                        continue
                    # ---
                    graph.add_edge(
                        node_symbol(atoms[centre_idx].symbol, centre_idx, (x, y, z)),
                        node_symbol(
                            atoms[nei_idx].symbol, nei_idx, (x + ox, y + oy, z + oz)
                        ),
                    )
                else:
                    ...

    # We need components that have at least one REAL atom
    molecule_graphs = [graph.subgraph(c) for c in nx.connected_components(graph)]

    molecules, molecule_groups = [], []
    for i, g in enumerate(molecule_graphs):
        is_real = any(d["is_real"] for u, d in g.nodes.data())
        if is_real:
            atomic_indices = tuple(sorted([d["index"] for u, d in g.nodes.data()]))
            if atomic_indices not in molecule_groups:
                molecule_groups.append(atomic_indices)
                molecules.append(MolecularAdsorbate.from_graph(atoms, g))

    return molecules


if __name__ == "__main__":
    ...
