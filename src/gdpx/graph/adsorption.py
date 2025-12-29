import networkx as nx
import numpy as np
from ase import Atom, Atoms
from ase.neighborlist import NeighborList, neighbor_list

from gdpx.group import evaluate_group_expression

from .base import NeighbourData
from .domain import build_domain_graph


def get_atop_sites(atoms: Atoms, graph: nx.Graph):
    """Get atop adsorption sites."""
    sites = []
    for node in graph.nodes:
        if node.shift != (0, 0, 0):
            continue
        a0 = atoms[node.idx]
        assert isinstance(a0, Atom)
        site_info = {
            "type": "atop",
            "atoms": (node,),
            "symbols": [a0.symbol],
            "position": a0.position,
        }
        sites.append(site_info)

    return sites


def get_bridge_sites(atoms: Atoms, graph: nx.Graph):
    """Get bridge adsorption sites."""
    box = atoms.get_cell()

    sites = []
    for u, v, _ in graph.edges(data=True):
        if u.shift != (0, 0, 0) and v.shift != (0, 0, 0):
            continue
        a0, a1 = atoms[u.idx], atoms[v.idx]
        assert isinstance(a0, Atom)
        assert isinstance(a1, Atom)
        p0 = a0.position + np.array(u.shift) @ box
        p1 = a1.position + np.array(v.shift) @ box
        direction = p1 - p0
        site_info = {
            "type": "bridge",
            "atoms": (u, v),
            "symbols": [a0.symbol, a1.symbol],
            "position": (p0 + p1) / 2,
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
        symbols, shifted_positions = [], []
        for node in nodes:
            a = atoms[node.idx]
            assert isinstance(a, Atom)
            p = a.position + np.array(node.shift) @ box
            shifted_positions.append(p)
            symbols.append(a.symbol)
        p0, p1, p2 = shifted_positions
        position = sum([p0, p1, p2], np.zeros(3)) / 3
        direction1 = p1 - p0  # i -> j
        direction2 = p2 - p0  # i -> k
        direction3 = p2 - p1  # j -> k
        site_info = {
            "type": "hollow",
            "atoms": tuple(nodes),
            "symbols": symbols,
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

    # Check indices to build graph
    num_atoms = len(atoms)
    group_indices = sorted(evaluate_group_expression(atoms, group_expr))

    # Build graph
    senders, receivers, distances, shifts = neighbor_list("ijdS", atoms, cutoff, self_interaction=True)
    neigh = NeighbourData(senders, receivers, distances, shifts)
    graph = build_domain_graph(atoms, neigh, group_indices)

    # Use neighbour data to find surface normals
    surf_normals = np.zeros((num_atoms, 3))
    for i in group_indices:
        masks = [idx == i for idx in neigh.senders]
        num_neighbors = np.sum(masks)
        if num_neighbors > 0:
            bond_vectors = []
            a_i = atoms[i]
            assert isinstance(a_i, Atom)
            for j, s in zip(neigh.receivers[masks], neigh.shifts[masks]):
                a_j = atoms[j]
                assert isinstance(a_j, Atom)
                vec = a_i.position - (a_j.position + s @ box)
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
