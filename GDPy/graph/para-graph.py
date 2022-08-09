#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import networkx as nx
import multiprocessing
import concurrent.futures
import pickle

from pathlib import Path

from joblib import Parallel, delayed

from ase.io import read, write

from GDPy.graph.utils import unique_chem_envs, compare_chem_envs, plot_graph

def new_unique_chem_envs(graph, chem_envs_groups, env_indices, verbose=False):
    """"""
    print("*** new org algo for chem_env cmp***")
    # Error checking, this should never really happen
    #print("chem_envs: ", chem_envs_groups)
    if len(chem_envs_groups) == 0:
        return [[],[]]

    # Keep track of known unique environments
    unique = []

    for index in env_indices:
        st = time.time()
        env = chem_envs_groups[index]
        for ug_idx, (unique_indices, ur_idx) in enumerate(unique):
            # check if already compared
            if graph.has_edge(index, ur_idx):
                continue
            # actual compare
            if compare_chem_envs(env, chem_envs_groups[ur_idx]):
                unique_indices.append(index)
                graph.add_edge(index, ur_idx, look_like=True)
                break
        else: # Was unique
            # add marker
            for (unique_indices, ur_idx) in unique:
                graph.add_edge(index, ur_idx, look_like=False)
            unique.append(([index], index))
        et = time.time()
        print("used time ", et-st, " for ", index, " for ", len(unique))
    
    # Zip trick to split into two lists to return
    # unique_envs, unique_groups
    # unique_groups = [[index,metadata],[]]
    return unique


def paragroup_unique_chem_envs(chem_envs_groups, metadata=None, n_jobs=1):
    nframes = len(chem_envs_groups)
    unique_graph = nx.Graph()
    for x in range(nframes):
        unique_graph.add_node(x)
    
    #unique = new_unique_chem_envs(unique_graph, chem_envs_groups, range(nframes))
    #removed_edges = []
    #for (u, v, d) in unique_graph.edges.data():
    #    if not d["look_like"]:
    #        removed_edges.append((u,v))
    #for u, v in removed_edges:
    #    unique_graph.remove_edge(u, v)
    #plot_graph(unique_graph, "xxx.png")


    if n_jobs > 1:
        st = time.time()
        # split data
        nframes = len(chem_envs_groups)
        size = nframes // n_jobs
        st_indices = [x*size for x in range(n_jobs)]
        end_indices = [x+size for x in st_indices]
        end_indices[-1] = nframes
        print(st_indices)
        print(end_indices)
        split_indices = []
        split_groups, split_metadata = [], []
        for s, e in zip(st_indices, end_indices):
            split_indices.append((s,e))
            split_groups.append(chem_envs_groups[s:e])
            split_metadata.append(list(range(s,e)))

        # run para cmp
        ret = Parallel(n_jobs=n_jobs)(delayed(new_unique_chem_envs)(unique_graph, chem_envs_groups, range(s,e)) for s, e in split_indices)
        # run final cmp
        env_indices = []
        for x in ret:
            env_indices.extend([a[1] for a in x])
        print(env_indices)
        print("nuniques: ", len(env_indices))
        unique = new_unique_chem_envs(unique_graph, chem_envs_groups, env_indices)

        et = time.time()
        print("split time: ", et - st)

        st = time.time()

        removed_edges = []
        for (u, v, d) in unique_graph.edges.data():
            if not d["look_like"]:
                removed_edges.append((u,v))
        for u, v in removed_edges:
            unique_graph.remove_edge(u, v)

        # merge results
        unique_group_graphs = [unique_graph.subgraph(c) for c in nx.connected_components(unique_graph)]

        unique_envs, unique_groups = [], []
        for g in unique_group_graphs:
            u_indices = sorted([u for u in g])
            unique_groups.append([metadata[u] for u in u_indices])
            unique_envs.append(chem_groups[u_indices[0]])
        else:
            unique_envs, unique_groups = unique_chem_envs(chem_envs_groups, metadata)

        et = time.time()
        print("graph time: ", et - st)

    return unique_envs, unique_groups

if __name__ == "__main__":
    #p = "/mnt/scratch2/users/40247882/oxides/graph/NewTest/Pt111/s44a/ParaCmpTest/t400/chemenvs-222.pkl"
    p = "/mnt/scratch2/users/40247882/oxides/graph/NewTest/Pt111/s44a/ParaCmpTest/t400/chemenvs.pkl"
    with open(p, "rb") as fopen:
        chem_groups = pickle.load(fopen)
    
    p = "/mnt/scratch2/users/40247882/oxides/graph/NewTest/Pt111/s44a/ParaCmpTest/t400/cands-400.xyz"
    frames = read(p, ":")
    nframes = len(frames)

    st = time.time()
    unique_envs, unique_groups = paragroup_unique_chem_envs(chem_groups, list(enumerate(frames)), 2)
    et = time.time()

    print("tot time: ", et-st)

    unique_data = []
    for i, x in enumerate(unique_groups):
        data = ["ug"+str(i)]
        data.extend([a[0] for a in x])
        unique_data.append(data)
    content = "# unique, indices\n"
    content += f"# ncandidates {nframes}\n"
    for d in unique_data:
        content += ("{:<8s}  "+"{:<8d}  "*(len(d)-1)+"\n").format(*d)

    with open(Path.cwd() / "unique-g.txt", "w") as fopen:
        fopen.write(content)

