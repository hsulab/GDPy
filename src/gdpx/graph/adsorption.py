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
            "symbols": [atoms[node.idx].symbol],
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
            "symbols": [atoms[u.idx].symbol, atoms[v.idx].symbol],
            "position": (atoms[u.idx].position + atoms[v.idx].position) / 2 + d["shift"] / 2,
            "direction": np.array([direction]),
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
        direction1 = (atoms[nodes[1].idx].position + np.array(nodes[1].shift) @ box) - (
            atoms[nodes[0].idx].position + np.array(nodes[0].shift) @ box
        )  # i -> j
        direction2 = (atoms[nodes[2].idx].position + np.array(nodes[2].shift) @ box) - (
            atoms[nodes[0].idx].position + np.array(nodes[0].shift) @ box
        )  # i -> k
        direction3 = (atoms[nodes[2].idx].position + np.array(nodes[2].shift) @ box) - (
            atoms[nodes[1].idx].position + np.array(nodes[1].shift) @ box
        )  # j -> k
        site_info = {
            "type": "hollow",
            "atoms": tuple(nodes),
            "symbols": [atoms[node.idx].symbol for node in nodes],
            "position": position,
            "direction": np.array([direction1, direction2, direction3]),
        }
        sites.append(site_info)

    return sites


def find_adsorption_sites_by_graph(
    atoms: Atoms,
    group_expr: str,
    cutoff: float = 3.0,
    max_order: int = 2,
    surf_index: int = 2,
    surf_normal_threshold: float = 1.0,
):
    """Find adsorption sites on a surface based on graph components.

    Args:
        atoms: The ASE Atoms object representing the surface.
        group_expr: The group expression to select surface atoms.
        cutoff: The cutoff distance to consider neighbors.
        max_order: The maximum order of sites, 0 for atop, 1 for bridge, 2 for hollow.
        surf_index: The index of the surface normal direction (0 for x, 1 for y, 2 for z).
        surf_normal_threshold: The threshold to determine surface normals.

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

    # Build neighbor list
    # We need self_interaction and bothways to determine surface normals but skip them in adding edges
    nl = NeighborList([cutoff / 2.0] * num_atoms, skin=0.0, self_interaction=True, bothways=True)
    nl.update(atoms)

    # Add edges
    use_edges = set()
    for i in group_indices:
        indices, offsets = nl.get_neighbors(i)
        for j, offset in zip(indices, offsets):
            edge_index = tuple(sorted((i, j)))
            if i == j or j not in group_indices or edge_index in use_edges:
                continue
            u = NodeID(int(i), canonicalize_shift((0, 0, 0)))
            v = NodeID(int(j), canonicalize_shift(offset))
            shift = offset @ box
            graph.add_edge(
                u,
                v,
                shift=shift,
                # distance=distance,
            )
            use_edges.add(edge_index)

    # Determin surface normals of selected atoms
    surf_normals = np.zeros((num_atoms, 3))
    for i in group_indices:
        indices, offsets = nl.get_neighbors(i)
        bond_vectors = []
        for j, o in zip(indices, offsets):
            shift = o @ box
            vec = atoms[i].position - (atoms[j].position + shift)
            bond_vectors.append(vec)
        s_vec = np.sum(bond_vectors, axis=0)
        s_norm = np.linalg.norm(s_vec)
        if s_norm > surf_normal_threshold:
            surf_normals[i] = s_vec / s_norm

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

    # Add normal information
    default_surface_normal = np.zeros(3)
    default_surface_normal[surf_index] = 1.0
    for site in sites:
        atom_indices = [node.idx for node in site["atoms"]]
        normal_vectors = [surf_normals[idx] for idx in atom_indices]
        avg_normal = np.mean(normal_vectors, axis=0)
        norm = np.linalg.norm(avg_normal)
        if norm > 1e-4:
            avg_normal /= norm
        else:  # fall back to default normal
            avg_normal = default_surface_normal
        site["normal"] = avg_normal

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
