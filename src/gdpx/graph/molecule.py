#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import itertools

from typing import List, Mapping

from ase import Atoms


def find_product(atoms: Atoms, reactants: List[List[int]], grid=[1,1,0], radii_multi=1.0, skin=0.0) -> List[List[int]]:
    """Find if there were a product from input reactants."""
    valid_indices = list(itertools.chain.from_iterable(reactants))

    # - create local graph
    covalent_radii = natural_cutoffs(atoms, radii_multi)
    nl = NeighborList(
        covalent_radii, 
        skin = skin, sorted=False,
        self_interaction=False, 
        bothways=True
    )
    nl.update(atoms)

    #print([covalent_radii[i] for i in valid_indices])

    graph = nx.Graph()
    
    #grid = [1,1,0] # for surface
    # -- add nodes
    for centre_idx in valid_indices:
        for x, y, z in grid_iterator(grid):
            graph.add_node(
                node_symbol(atoms[centre_idx].symbol, centre_idx, (x,y,z)),
                index=centre_idx
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
                        node_symbol(atoms[centre_idx].symbol, centre_idx, (x,y,z)),
                        node_symbol(atoms[nei_idx].symbol, nei_idx, (x+ox,y+oy,z+oz))
                    )
                else:
                    ...
    
    #plot_graph(graph, "xxx.png")

    # - find products
    reax_nodes = [node_symbol(atoms[i].symbol, i, (0,0,0)) for i in valid_indices]
    reax_graphs = nx.subgraph(graph, reax_nodes)

    prod_graphs = [reax_graphs.subgraph(c) for c in nx.connected_components(reax_graphs)]

    products = [[unpack_node_name(u)[1] for u in g.nodes()] for g in prod_graphs]

    return products

from ase.formula import Formula
def find_molecules(atoms: Atoms, valid_indices: List[int], grid=[1,1,0], radii_multi=1.0, skin=0.0) -> Mapping[str,List[List[int]]]:
    """Find if there were a product from input reactants."""
    #valid_indices = list(chain.from_iterable(reactants))

    # - create local graph
    covalent_radii = natural_cutoffs(atoms, radii_multi)
    nl = NeighborList(
        covalent_radii, 
        skin = skin, sorted=False,
        self_interaction=False, 
        bothways=True
    )
    nl.update(atoms)

    #print([covalent_radii[i] for i in valid_indices])

    graph = nx.Graph()
    
    #grid = [1,1,0] # for surface
    # -- add nodes
    for centre_idx in valid_indices:
        for x, y, z in grid_iterator(grid):
            graph.add_node(
                node_symbol(atoms[centre_idx].symbol, centre_idx, (x,y,z)),
                index=centre_idx
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
                        node_symbol(atoms[centre_idx].symbol, centre_idx, (x,y,z)),
                        node_symbol(atoms[nei_idx].symbol, nei_idx, (x+ox,y+oy,z+oz))
                    )
                else:
                    ...
    
    #plot_graph(graph, "xxx.png")

    # - find products
    reax_nodes = [node_symbol(atoms[i].symbol, i, (0,0,0)) for i in valid_indices]
    reax_graphs = nx.subgraph(graph, reax_nodes)

    prod_graphs = [reax_graphs.subgraph(c) for c in nx.connected_components(reax_graphs)]

    products = [[unpack_node_name(u)[1] for u in g.nodes()] for g in prod_graphs]

    # - get formula
    fragments = {}
    for atomic_indices in products:
        symbols = [atoms[i].symbol for i in atomic_indices]
        formula = Formula.from_list(symbols).format("hill")
        if formula in fragments:
            fragments[formula].append(atomic_indices)
        else:
            fragments[formula] = [atomic_indices]

    return fragments 


if __name__ == "__main__":
    ...