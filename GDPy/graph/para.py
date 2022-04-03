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

def new_unique_chem_envs(extra_info, chem_envs_groups, env_indices, verbose=False):
    """"""
    if verbose:
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

        # NOTE: use a flag since there may continue all and not break
        found_similar = False
        for ug_idx, (unique_indices, ur_idx) in enumerate(unique):
            # check if already compared
            if ur_idx in extra_info[index]["others"]:
                continue
            # add reclusive
            extra_info[index]["others"].append(ur_idx)
            extra_info[ur_idx]["others"].append(index)
            # actual compare
            if compare_chem_envs(env, chem_envs_groups[ur_idx]):
                found_similar = True
                unique_indices.append(index)
                extra_info[ur_idx]["similar"].append(index)
                #print(ur_idx, extra_info[ur_idx])
                if len(extra_info[index]["similar"]) > 0:
                    extra_info[ur_idx]["similar"].extend(extra_info[index]["similar"])
                    extra_info[index]["similar"] = []
                break

        if not found_similar: # Was unique
            unique.append(([index], index))

        et = time.time()
        if verbose:
            print("used time ", et-st, " for ", index, " for ", len(unique))
    
    return unique

def get_indices(env_indices, n_jobs):
    nframes = len(env_indices)
    size = nframes // n_jobs

    st_indices = [x*size for x in range(n_jobs)]
    end_indices = [x+size for x in st_indices]
    end_indices[-1] = nframes

    env_indices_splits = []
    for s, e in zip(st_indices, end_indices):
        env_indices_splits.append([env_indices[x] for x in range(s,e)])

    return env_indices_splits

def get_groups(extra_info, env_indices, nsplits):
    """"""
    nframes = len(env_indices)
    size = nframes // nsplits
    st_indices = [x*size for x in range(nsplits)]
    end_indices = [x+size for x in st_indices]
    end_indices[-1] = nframes

    temp_env_indices = env_indices.copy()
    env_indices_splits = []

    nvalid = 0
    while nvalid != nsplits-1:
        if len(temp_env_indices) < 1:
            break
        s, e = st_indices[nvalid], end_indices[nvalid]
        size = e - s
        #print("size: ", size)
        #print(len(temp_env_indices), temp_env_indices)
        first = temp_env_indices[0]
        del temp_env_indices[0]
        nrest = len(temp_env_indices)
        #mates = [x for x in temp_env_indices if x not in extra_info[first]["others"]]
        mates = []
        count = 0
        while len(mates) < size-1:
            if count >= nrest:
                break
            if temp_env_indices[count] not in extra_info[first]["others"]:
                mates.append(temp_env_indices[count])
            count += 1
        if len(mates) < 1:
            continue
        env_indices_splits.append([first]+mates)
        temp_env_indices = list(set(temp_env_indices).difference(mates))
        nvalid += 1

    # last group
    if len(temp_env_indices) > 0:
        env_indices_splits.append(temp_env_indices)

    return env_indices_splits

def merge_results(extra_info, chem_envs_groups, metadata):
    duplicates = []
    for i, info in enumerate(extra_info):
        if len(info["similar"]) > 0:
            duplicates.extend(info["similar"])

    unique_indices = []
    unique_envs, unique_groups = [], []
    for i, info in enumerate(extra_info):
        if i in duplicates:
            continue
        unique_indices.append(i)
        unique_envs.append(chem_envs_groups[i])
        u_indices = [i]
        if len(info["similar"]) > 0:
            u_indices += info["similar"]
        u_indices = sorted(u_indices)
        unique_groups.append([metadata[u] for u in u_indices])
    # TODO: resort

    return unique_indices, unique_envs, unique_groups


def paragroup_unique_chem_envs(chem_envs_groups, metadata=None, n_jobs=1):
    """"""
    nframes = len(chem_envs_groups)

    saved_info = Path("extra_info.pkl")
    if saved_info.exists():
        with open(saved_info, "rb") as fopen:
            extra_info = pickle.load(fopen)
        env_indices, unique_envs, unique_groups = merge_results(extra_info, chem_envs_groups, metadata)
        print("load cached comparasion info...")
    else:
        extra_info = [
            {"similar": [], "others": []} for i in range(nframes)
        ]
        env_indices = list(range(nframes))

    # determine atomic task size
    # NOTE: MAYBE A BUG? 400/64 may miss some comparasions
    # that is for too small atomic task
    #nsplits = 64
    nsplits = n_jobs*(nframes // 16 // n_jobs)
    print("nsplits: ", nsplits)
    
    with Parallel(
        n_jobs=n_jobs, 
        prefer="threads",
        #require="sharedmem"
    ) as para:
        st = time.time()
        # split data

        for icount in range(500):
            print(f"--- {icount} ---")
            print("input nuniques: ", len(env_indices))
            #print("nsplits: ", nsplits)
            #env_indices_splits = get_indices(env_indices, nsplits)
            env_indices_splits = get_groups(extra_info, env_indices, nsplits)
            print(env_indices_splits)

            # run para cmp
            ret = para(delayed(new_unique_chem_envs)(extra_info, chem_envs_groups, env_indices) for env_indices in env_indices_splits)

            et = time.time()
            print("split ", nsplits, " for count ", icount, " time: ", et - st)

            # run final cmp
            env_indices = []
            for x in ret:
                env_indices.extend([a[1] for a in x])
            env_indices = sorted(env_indices)
            print("output nuniques: ", len(env_indices))

            #print("--- extra info ---")
            #for i, x in enumerate(extra_info):
            #    if i in env_indices:
            #        print(i, sorted(x["others"]))
            
            # check diff
            temp_env_indices = env_indices.copy()
            for e_idx in temp_env_indices:
                compared_others = sorted([e_idx] + extra_info[e_idx]["others"])
                if set(compared_others).intersection(set(temp_env_indices)) != set(temp_env_indices):
                    break
            else:
                print("finished...")
                break
                
            if (icount+1) % 50 == 0:
                print("save temp comparasion info...")
                with open(saved_info.name+"-"+str(icount+1), "wb") as fopen:
                    pickle.dump(extra_info, fopen)
            
        #unique = new_unique_chem_envs(extra_info, chem_envs_groups, env_indices)
        #et = time.time()
        #print("complete time: ", et - st)
    
    with open(saved_info, "wb") as fopen:
        pickle.dump(extra_info, fopen)

    st = time.time()

    unique_indices, unique_envs, unique_groups = merge_results(extra_info, chem_envs_groups, metadata)
        
    et = time.time()
    print("merge data time: ", et - st)

    return unique_envs, unique_groups

if __name__ == "__main__":
    #p = "/mnt/scratch2/users/40247882/oxides/graph/NewTest/Pt111/s44a/ParaCmpTest/t400/chemenvs-222.pkl"
    #p = "/mnt/scratch2/users/40247882/oxides/graph/NewTest/Pt111/s44a/ParaCmpTest/t20/chemenvs.pkl"
    #p = "/mnt/scratch2/users/40247882/oxides/graph/NewTest/Pt111/s44a/ParaCmpTest/t100/chemenvs.pkl"
    p = "/mnt/scratch2/users/40247882/oxides/graph/NewTest/Pt111/s44a/ParaCmpTest/t400/chemenvs.pkl"
    with open(p, "rb") as fopen:
        chem_groups = pickle.load(fopen)
    
    #p = "/mnt/scratch2/users/40247882/oxides/graph/NewTest/Pt111/s44a/ParaCmpTest/t20/cands-20.xyz"
    #p = "/mnt/scratch2/users/40247882/oxides/graph/NewTest/Pt111/s44a/ParaCmpTest/t100/cands-100.xyz"
    p = "/mnt/scratch2/users/40247882/oxides/graph/NewTest/Pt111/s44a/ParaCmpTest/t400/cands-400.xyz"
    frames = read(p, ":")
    nframes = len(frames)

    st = time.time()
    unique_envs, unique_groups = paragroup_unique_chem_envs(chem_groups, list(enumerate(frames)), 16)
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