#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from typing import NamedTuple

import networkx as nx
import numpy as np
from ase import Atoms
from ase.neighborlist import NeighborList, neighbor_list

from gdpx.group import evaluate_group_expression

from .utils import grid_iterator


class NodeID(NamedTuple):

    idx: int
    shift: tuple[int, int, int]


def canonicalize_shift(shift: np.ndarray | tuple[int, int, int]) -> tuple[int, int, int]:
    """
    Convert any integer-like iterable (possibly np.int32 or np.int64) into a tuple of Python ints.
    This ensures consistent hashing and equality in NetworkX.
    """
    return tuple(int(x) for x in np.asarray(shift, dtype=np.int32))


def get_atop_sites(atoms: Atoms, graph: nx.Graph):
    """Get atop adsorption sites."""
    sites = []
    for node in graph.nodes:
        if node.shift != (0, 0, 0):
            continue
        site_info = {
            "type": "atop",
            "atoms": (node,),
            "position": atoms[node.idx].position,
        }
        sites.append(site_info)

    return sites


def get_bridge_sites(atoms: Atoms, graph: nx.Graph):
    """Get bridge adsorption sites."""
    sites = []
    for u, v, d in graph.edges(data=True):
        if u.shift != (0, 0, 0) and v.shift != (0, 0, 0):
            continue
        direction = (atoms[v.idx].position + d["shift"]) - atoms[u.idx].position
        site_info = {
            "type": "bridge",
            "atoms": (u, v),
            "position": (atoms[u.idx].position + atoms[v.idx].position) / 2 + d["shift"] / 2,
            "direction": direction,
        }
        sites.append(site_info)

    return sites


def get_hollow_sites(atoms: Atoms, graph: nx.Graph):
    """Get hollow adsorption sites."""
    box = atoms.get_cell()

    sites = []
    for component in nx.connected_components(graph):
        if len(component) != 3:
            continue
        nodes = list(component)
        shifts = [node.shift for node in nodes]
        if not (shifts.count((0, 0, 0)) > 0):
            continue
        position = (
            sum(
                (atoms[node.idx].position + np.array(node.shift) @ box for node in nodes),
                np.zeros(3),
            )
            / 3
        )
        site_info = {
            "type": "hollow",
            "atoms": tuple(nodes),
            "position": position,
        }
        sites.append(site_info)

    return sites


def find_adsorption_sites_by_graph(
    atoms: Atoms, group_expr: str, cutoff: float = 3.0, max_order: int = 2, surf_index: int = 2
):
    """Find adsorption sites on a surface based on graph components.

    Args:
        atoms: The ASE Atoms object representing the surface.
        group_expr: The group expression to select surface atoms.
        cutoff: The cutoff distance to consider neighbors.
        max_order: The maximum order of sites, 0 for atop, 1 for bridge, 2 for hollow.
        surf_index: The index of the surface normal direction (0 for x, 1 for y, 2 for z).

    """
    graph = nx.Graph()

    # Get the maximum grid range
    box = atoms.get_cell()
    lengths = box.lengths()
    max_grid = [int((cutoff // l) + 1) for l in lengths]
    max_grid[surf_index] = 0  # No need to search in the surface normal direction
    grids = list(grid_iterator(tuple(max_grid)))

    # Check indices to build graph
    num_atoms = len(atoms)
    group_indices = sorted(evaluate_group_expression(atoms, group_expr))

    # Add nodes
    for grid in grids:
        for idx in group_indices:
            graph.add_node(
                NodeID(int(idx), canonicalize_shift(grid)),
            )

    # Add edges
    nl = NeighborList([cutoff / 2.0] * num_atoms, self_interaction=False, bothways=False)
    nl.update(atoms)

    for i in group_indices:
        indices, offsets = nl.get_neighbors(i)
        for j, offset in zip(indices, offsets):
            if j not in group_indices:
                continue
            u = NodeID(int(i), canonicalize_shift((0, 0, 0)))
            v = NodeID(int(j), canonicalize_shift(offset))
            shift = offset @ box
            # distance = np.linalg.norm(atoms[i].position - (atoms[j].position + shift))
            graph.add_edge(
                u,
                v,
                shift=shift,
                # distance=distance,
            )

    print(f"{num_atoms=}")
    print(f"{graph.number_of_nodes()=} , {graph.number_of_edges()=}")

    # Find sites
    sites = []
    if max_order >= 0:
        atop_sites = get_atop_sites(atoms, graph)
        sites.extend(atop_sites)

    if max_order >= 1:
        bridge_sites = get_bridge_sites(atoms, graph)
        sites.extend(bridge_sites)

    if max_order >= 2:
        hollow_sites = get_hollow_sites(atoms, graph)
        sites.extend(hollow_sites)

    return sites


def find_adsorption_sites_by_pair(
    atoms: Atoms,
    group_expr: str,
    cutoff: float = 3.0,
    max_order: int = 2,
    surf_index: int = 2,
):
    """Find adsorption sites on a surface."""
    num_atoms = len(atoms)
    box = atoms.get_cell()

    selected_indices = sorted(evaluate_group_expression(atoms, group_expr))

    nlist = NeighborList([cutoff / 2.0] * num_atoms, self_interaction=False, bothways=False)
    nlist.update(atoms)

    pairs = []
    for i in selected_indices:
        n_indices, n_offsets = nlist.get_neighbors(i)
        for j, o in zip(n_indices, n_offsets):
            if j in selected_indices:
                pairs.append(((i, j), o))

    sites = []
    for (i, j), o in pairs:
        pos_i = atoms.positions[i]
        pos_j = atoms.positions[j] + np.dot(o, box)
        site_position = (pos_i + pos_j) / 2
        site_direction = pos_j - pos_i
        sites.append(
            {
                "atoms": (i, j),
                "position": site_position,
                "direction": site_direction,
            }
        )

    return sites


if __name__ == "__main__":
    ...
