#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
import shutil
import argparse
from pathlib import Path

import numpy as np
import networkx as nx

from tqdm import tqdm

from ase import Atoms
from ase.io import read, write


"""This temporarily stores some utilities for atoms object.
"""

def check_convergence(atoms, fmax=0.05):
    """Check the convergence of the trajectory"""

    forces = atoms.get_forces()

    max_force = np.max(np.fabs(forces))

    converged = False
    if max_force < fmax:
        converged = True 

    return converged

def merge_xyz():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p", "--pattern",
        help="pattern to find xyz files"
    )
    args = parser.parse_args()

    #pattern = "s2hcp"
    pattern = args.pattern
    cwd = Path.cwd()

    sorted_paths = []
    for p in cwd.glob(pattern+"*.xyz"):
        sorted_paths.append(p)
    sorted_paths.sort()

    frames = []
    for p in sorted_paths:
        cur_frames = read(p, ":")
        print(p.stem)
        for atoms in cur_frames:
            atoms.info["source"] = str(p.stem)
        print(p, " #frames: ", len(cur_frames))
        frames.extend(cur_frames)
    print("TOTAL #frames: ", len(frames))
    write(pattern+"-0906.xyz", frames)

    for p in sorted_paths:
        shutil.move(p, cwd / ("bak."+p.name))


def sort_atoms(atoms):
    # sort atoms by symbols and z-positions especially for supercells 
    numbers = atoms.numbers 
    zposes = atoms.positions[:,2].tolist()
    sorted_indices = np.lexsort((zposes,numbers))
    atoms = atoms[sorted_indices]

    return atoms


def try_sort():
    frames = read('./test_data.xyz', ':')
    
    new_frames = [sort_atoms(atoms) for atoms in frames]
    write('new_test.xyz', new_frames)


def get_structure_type(atoms_: Atoms) -> str:
    """Determine the structure type based on atom connectivity.

    For bulk, there are bond connections in xyz 3 dimensions. For surface, there 
    are bond connections in one dimensions (default should be z-axis). For cluster
    or molecue, there are no bond connections among neighbour cells.

    Returns:
        bulk, surface, and cluster

    """
    #natoms = len(atoms)

    #from ase.neighborlist import natural_cutoffs, NeighborList
    #cutoffs = natural_cutoffs(atoms, mult=1.0)
    #neigh = NeighborList(
    #    cutoffs, skin=0.0, sorted=False, self_interaction=True, bothways=False,
    #)
    #neigh.update(atoms)

    #stype = "cluster"
    #has_x, has_y, has_z = False, False, False
    #for i in range(natoms):
    #    indices, offsets = neigh.get_neighbors(i)
    #    for j, offset in zip(indices, offsets):
    #        if offset[0] != 0:
    #            has_x = True
    #        if offset[1] != 0:
    #            has_y = True
    #        if offset[2] != 0:
    #            has_z = True
    #        print(f"{i}-{j}: ", offset, [has_x, has_y, has_z])
    #    if has_x and has_y and has_z:
    #        stype = "bulk"
    #        break
    #if sum([has_x, has_y, has_z]) == 2:
    #    stype = "surface"
    #    # TODO: find in which direction is the vaccum
    #    ...
    #print([has_x, has_y, has_z])
    atoms = copy.deepcopy(atoms_)
    #stype = atoms.info.get("stype")

    # -
    graph = create_multipgraph(atoms)
    # NOTE: A cluster has atoms concentrated around the centre.
    is_cluster = True
    # TODO: water box?
    #print(graph)
    for u, v, d in graph.edges.data():
        #print(u, v, d)
        # NOTE: at least a surface if there are bonds between neighbour boxs
        # TODO: a single molecule crosses the boundary?
        if sum(d["offset"]) != 0:
            is_cluster = False
            break

    if not is_cluster:
        # --- check if surface or bulk
        graph = create_multipgraph(atoms, pbc=False)
        subgraphs = [graph.subgraph(nodes) for nodes in nx.connected_components(graph)]
        #print(subgraphs)
        # TODO: water box?
        cell_ = atoms.get_cell(complete=True)
        if len(subgraphs) == 1:
            #stype = "bulk"
            positions_ = atoms.get_positions()
        else:
            cops = [
                np.average(atoms.positions[g.nodes], axis=0).tolist()
                for g in subgraphs
            ]
            orders = [x[0] for x in sorted(enumerate(cops), key=lambda x: x[1][2])]
            #print(cops)
            #print(orders)
            positions_ = copy.deepcopy(atoms.get_positions())
            for i in orders[1:]:
                for a_idx in subgraphs[i].nodes:
                    positions_[a_idx] += np.dot(cell_, [0,0,-1])
            # NOTE: avoid some bulk systems that they are moved far below
            if np.average(positions_, axis=0)[2] < 0.:
                positions_ = copy.deepcopy(atoms.get_positions())

        zvec = np.cross(cell_[0], cell_[1])
        zlength = np.dot(zvec/np.linalg.norm(zvec), cell_[2])
        max_zpos = np.max(positions_[:,2])
        #print("max_zpos: ", max_zpos)
        if max_zpos + 6.0 <= zlength: # NOTE: larger than a cutoff
            stype = "surface"
        else:
            stype = "bulk"
        #print(stype)
    else:
        stype = "cluster"

    return stype

def create_multipgraph(atoms_, pbc=True):
    """"""
    atoms = copy.deepcopy(atoms_)

    graph = nx.MultiGraph(atoms=atoms)
    #print(graph.graph)

    # - find neighbours
    natoms = len(atoms)
    atoms.pbc = pbc
    #atoms.pbc = False
    #atoms.pbc = [True, True, False]

    from ase.neighborlist import natural_cutoffs, NeighborList
    cutoffs = natural_cutoffs(atoms, mult=1.2)
    #cutoffs = [6.0]*natoms
    neigh = NeighborList(
        cutoffs, skin=0.0, sorted=False, self_interaction=False, bothways=False,
    )
    neigh.update(atoms)

    # -- add nodes
    nodes = [[i, {"symbol": a.symbol}] for i, a in enumerate(atoms)]
    graph.add_nodes_from(nodes)
    #print(graph)
    #for u, d in graph.nodes.data():
    #    print(u, d)

    # -- add edges
    edges = []
    for i in range(natoms):
        indices, offsets = neigh.get_neighbors(i)
        for j, offset in zip(indices, offsets):
            edges.append([i, j, {"route": 2, "offset": offset.tolist()}])
    graph.add_edges_from(edges)

    #print("--- edges ---") # find egdes without offset [0,0,0]
    #for u, v, d in graph.edges.data():
    #    print(u, v, d)
    #for u, v, keys in graph.edges(keys=True):
    #    print(u, v, keys)
    #for n, nbrsdict in graph.adjacency():
    #    print(n, nbrsdict)
    #    for nbr, keydict in nbrsdict.items():
    #        print(nbr, keydict)
    #        for key, eattr in keydict.items():
    #            ...
    #print(graph.edges())
    #print(set(graph.edges()))
    #for u, v, d in graph.edges.data(nbunch=[10]):
    #    print(u, v, d)
    
    # -- find subgraphs
    #subgraphs = nx.connected_components(graph)
    #print(subgraphs)

    return graph


if __name__ == "__main__":
    """"""
    stru_dict = {}

    frames = read("/home/jx1279/projects/sintering/dataset/Cu12.xyz", ":")
    #frames = read("/home/jx1279/projects/sintering/dataset/Cu16.xyz", ":")
    #frames = read("/home/jx1279/projects/sintering/dataset/Cu87.xyz", ":")
    nframes = len(frames)
    print("nframes: ", nframes)

    for i, atoms in tqdm(enumerate(frames)):
        atoms.info["confid"] = i
        # NOTE: cluster/molecule (0d), nanowire (1d), surface (2d), bulk (3d)
        # TODO: determine by volume?
        stype = get_structure_type(atoms)

        # -- 
        name = f"{atoms.get_chemical_formula()}_{stype}"
        if name in stru_dict:
            stru_dict[name].append(atoms)
        else:
            stru_dict[name] = [atoms]
    
    new_nframes = 0
    for (name, cur_frames) in stru_dict.items():
        new_nframes += len(cur_frames)
        print(len(cur_frames))
        write(f"./{name}.xyz", cur_frames)
    print(new_nframes)
    ...
